from pythran.analyses.aliases import Aliases
from pythran.analyses.argument_effects import ArgumentEffects
from pythran.analyses.identifiers import Identifiers
from pythran.analyses.pure_expressions import PureExpressions
from pythran.passmanager import FunctionAnalysis
from pythran.syntax import PythranSyntaxError
from pythran.utils import get_variable, isattr
import pythran.metadata as md
import pythran.openmp as openmp
import gast as ast
import sys
class LazynessAnalysis(FunctionAnalysis):
    """
    Returns number of time a name is used.

    +inf if it is use in a
    loop, if a variable used to compute it is modify before
    its last use or if it is use in a function call (as it is not an
    interprocedural analysis)

    >>> import gast as ast, sys
    >>> from pythran import passmanager, backend
    >>> code = "def foo(): c = 1; a = c + 2; c = 2; b = c + c + a; return b"
    >>> node = ast.parse(code)
    >>> pm = passmanager.PassManager("test")
    >>> res = pm.gather(LazynessAnalysis, node)
    >>> res['a'], res['b'], res['c']
    (inf, 1, 2)
    >>> code = '''
    ... def foo():
    ...     k = 2
    ...     for i in [1, 2]:
    ...         builtins.print(k)
    ...         k = i
    ...     builtins.print(k)'''
    >>> node = ast.parse(code)
    >>> res = pm.gather(LazynessAnalysis, node)
    >>> (res['i'], res['k']) == (sys.maxsize, 1)
    True
    >>> code = '''
    ... def foo():
    ...     k = 2
    ...     for i in [1, 2]:
    ...         builtins.print(k)
    ...         k = i
    ...         builtins.print(k)'''
    >>> node = ast.parse(code)
    >>> res = pm.gather(LazynessAnalysis, node)
    >>> (res['i'], res['k']) == (sys.maxsize, 2)
    True
    >>> code = '''
    ... def foo():
    ...     d = 0
    ...     for i in [0, 1]:
    ...         for j in [0, 1]:
    ...             k = 1
    ...             d += k * 2
    ...     return d'''
    >>> node = ast.parse(code)
    >>> res = pm.gather(LazynessAnalysis, node)
    >>> res['k']
    1
    >>> code = '''
    ... def foo():
    ...     k = 2
    ...     for i in [1, 2]:
    ...         builtins.print(k)'''
    >>> node = ast.parse(code)
    >>> res = pm.gather(LazynessAnalysis, node)
    >>> res['k'] == sys.maxsize
    True
    >>> code = '''
    ... def foo():
    ...     k = builtins.sum
    ...     builtins.print(k([1, 2]))'''
    >>> node = ast.parse(code)
    >>> res = pm.gather(LazynessAnalysis, node)
    >>> res['k']
    1
    """
    INF = float('inf')
    MANY = sys.maxsize

    def __init__(self):
        self.result = dict()
        self.name_count = dict()
        self.use = dict()
        self.dead = set()
        self.pre_loop_count = dict()
        self.in_omp = set()
        self.name_to_nodes = dict()
        super(LazynessAnalysis, self).__init__(ArgumentEffects, Aliases, PureExpressions)

    def modify(self, name):
        dead_vars = [var for var, deps in self.use.items() if name in deps]
        self.dead.update(dead_vars)
        for var in dead_vars:
            dead_aliases = [alias.id for alias in self.name_to_nodes[var] if isinstance(alias, ast.Name)]
            self.dead.update(dead_aliases)

    def assign_to(self, node, from_):
        if isinstance(node, ast.Name):
            self.name_to_nodes.setdefault(node.id, set()).add(node)
        if node.id in self.dead:
            self.dead.remove(node.id)
        self.result[node.id] = max(self.result.get(node.id, 0), self.name_count.get(node.id, 0))
        self.in_omp.discard(node.id)
        pre_loop = self.pre_loop_count.setdefault(node.id, (0, True))
        if not pre_loop[1]:
            self.pre_loop_count[node.id] = (pre_loop[0], True)
        self.modify(node.id)
        self.name_count[node.id] = 0
        self.use[node.id] = set(from_)

    def visit(self, node):
        old_omp = self.in_omp
        omp_nodes = md.get(node, openmp.OMPDirective)
        if omp_nodes:
            self.in_omp = set(self.name_count.keys())
        super(LazynessAnalysis, self).visit(node)
        if omp_nodes:
            new_nodes = set(self.name_count).difference(self.in_omp)
            for omp_node in omp_nodes:
                for n in omp_node.deps:
                    if isinstance(n, ast.Name):
                        self.result[n.id] = LazynessAnalysis.INF
            self.dead.update(new_nodes)
        self.in_omp = old_omp

    def visit_FunctionDef(self, node):
        self.ids = self.gather(Identifiers, node)
        self.generic_visit(node)

    def visit_Assign(self, node):
        md.visit(self, node)
        self.visit(node.value)
        ids = self.gather(Identifiers, node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assign_to(target, ids)
                if node.value not in self.pure_expressions:
                    self.result[target.id] = LazynessAnalysis.INF
            elif isinstance(target, ast.Subscript) or isattr(target):
                var_name = get_variable(target)
                if isinstance(var_name, ast.Name):
                    self.modify(var_name.id)
                    self.result[var_name.id] = LazynessAnalysis.INF
            else:
                raise PythranSyntaxError('Assign to unknown node', node)

    def visit_AugAssign(self, node):
        md.visit(self, node)
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.modify(node.target.id)
            self.result[node.target.id] = LazynessAnalysis.INF
        elif isinstance(node.target, ast.Subscript) or isattr(node.target):
            var_name = get_variable(node.target)
            self.modify(var_name.id)
            self.result[var_name.id] = LazynessAnalysis.INF
        else:
            raise PythranSyntaxError('AugAssign to unknown node', node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.use:

            def is_loc_var(x):
                return isinstance(x, ast.Name) and x.id in self.ids
            alias_names = [var for var in self.aliases[node] if is_loc_var(var)]
            alias_names = {x.id for x in alias_names}
            alias_names.add(node.id)
            for alias in alias_names:
                if node.id in self.dead or node.id in self.in_omp:
                    self.result[alias] = LazynessAnalysis.INF
                elif alias in self.name_count:
                    self.name_count[alias] += 1
                    pre_loop = self.pre_loop_count.setdefault(alias, (0, False))
                    if not pre_loop[1]:
                        self.pre_loop_count[alias] = (pre_loop[0] + 1, False)
                else:
                    pass
        elif isinstance(node.ctx, ast.Param):
            self.name_count[node.id] = 0
            self.use[node.id] = set()
        elif isinstance(node.ctx, ast.Store):
            self.name_count[node.id] = LazynessAnalysis.INF
            self.use[node.id] = set()
        else:
            pass

    def visit_If(self, node):
        md.visit(self, node)
        self.visit(node.test)
        old_count = dict(self.name_count)
        old_dead = set(self.dead)
        old_deps = {a: set(b) for a, b in self.use.items()}
        body = node.body if isinstance(node.body, list) else [node.body]
        for stmt in body:
            self.visit(stmt)
        mid_count = self.name_count
        mid_dead = self.dead
        mid_deps = self.use
        self.name_count = old_count
        self.dead = old_dead
        self.use = old_deps
        orelse = node.orelse if isinstance(node.orelse, list) else [node.orelse]
        for stmt in orelse:
            self.visit(stmt)
        for key in self.use:
            if key in mid_deps:
                self.use[key].update(mid_deps[key])
        for key in mid_deps:
            if key not in self.use:
                self.use[key] = set(mid_deps[key])
        names = set(self.name_count.keys()).union(mid_count.keys())
        for name in names:
            val_body = mid_count.get(name, 0)
            val_else = self.name_count.get(name, 0)
            self.name_count[name] = max(val_body, val_else)
        self.dead.update(mid_dead)
    visit_IfExp = visit_If

    def visit_loop(self, body):
        old_pre_count = self.pre_loop_count
        self.pre_loop_count = dict()
        for stmt in body:
            self.visit(stmt)
        no_assign = [n for n, (_, a) in self.pre_loop_count.items() if not a]
        self.result.update(zip(no_assign, [LazynessAnalysis.MANY] * len(no_assign)))
        for k, v in self.pre_loop_count.items():
            loop_value = v[0] + self.name_count[k]
            self.result[k] = max(self.result.get(k, 0), loop_value)
        dead = self.dead.intersection(self.pre_loop_count)
        self.result.update(zip(dead, [LazynessAnalysis.INF] * len(dead)))
        for k, v in old_pre_count.items():
            if v[1] or k not in self.pre_loop_count:
                self.pre_loop_count[k] = v
            else:
                self.pre_loop_count[k] = (v[0] + self.pre_loop_count[k][0], self.pre_loop_count[k][1])

    def visit_For(self, node):
        md.visit(self, node)
        ids = self.gather(Identifiers, node.iter)
        if isinstance(node.target, ast.Name):
            self.assign_to(node.target, ids)
            self.result[node.target.id] = LazynessAnalysis.INF
        else:
            err = 'Assignation in for loop not to a Name'
            raise PythranSyntaxError(err, node)
        self.visit_loop(node.body)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node):
        md.visit(self, node)
        self.visit(node.test)
        self.visit_loop(node.body)
        for stmt in node.orelse:
            self.visit(stmt)

    def func_args_lazyness(self, func_name, args, node):
        for fun in self.aliases[func_name]:
            if isinstance(fun, ast.Call):
                self.func_args_lazyness(fun.args[0], fun.args[1:] + args, node)
            elif fun in self.argument_effects:
                for i, arg in enumerate(self.argument_effects[fun]):
                    if arg and len(args) > i:
                        if isinstance(args[i], ast.Name):
                            self.modify(args[i].id)
            elif isinstance(fun, ast.Name):
                continue
            else:
                for arg in args:
                    self.modify(arg)

    def visit_Call(self, node):
        """
        Compute use of variables in a function call.

        Each arg is use once and function name too.
        Information about modified arguments is forwarded to
        func_args_lazyness.
        """
        md.visit(self, node)
        for arg in node.args:
            self.visit(arg)
        self.func_args_lazyness(node.func, node.args, node)
        self.visit(node.func)

    def run(self, node):
        result = super(LazynessAnalysis, self).run(node)
        for name, val in self.name_count.items():
            old_val = result.get(name, 0)
            result[name] = max(old_val, val)
        self.result = result
        return self.result