import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
class RangeValuesSimple(RangeValuesBase):
    """
    This analyse extract positive subscripts from code.

    It is flow sensitive and aliasing is not taken into account as integer
    doesn't create aliasing in Python.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse('''
    ... def foo(a):
    ...     for i in builtins.range(1, 10):
    ...         c = i // 2''')
    >>> pm = passmanager.PassManager("test")
    >>> res = pm.gather(RangeValuesSimple, node)
    >>> res['c'], res['i']
    (Interval(low=0, high=5), Interval(low=1, high=10))
    """

    def __init__(self, parent=None):
        if parent is not None:
            self.parent = parent
            self.ctx = parent.ctx
            self.deps = parent.deps
            self.result = parent.result
            self.aliases = parent.aliases
            self.passmanager = parent.passmanager
        else:
            super(RangeValuesSimple, self).__init__()

    def generic_visit(self, node):
        """ Other nodes are not known and range value neither. """
        super(RangeValuesSimple, self).generic_visit(node)
        return self.add(node, UNKNOWN_RANGE)

    def save_state(self):
        return (self.aliases,)

    def restore_state(self, state):
        self.aliases, = state

    def function_visitor(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Return(self, node):
        if node.value:
            return_range = self.visit(node.value)
            return self.add(RangeValues.ResultHolder, return_range)
        else:
            return self.generic_visit(node)

    def visit_Assert(self, node):
        """
        Constraint the range of variables

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(a): assert a >= 1; b = a + 1")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['a']
        Interval(low=1, high=inf)
        >>> res['b']
        Interval(low=2, high=inf)
        """
        self.generic_visit(node)
        bound_range(self.result, self.aliases, node.test)

    def visit_Assign(self, node):
        """
        Set range value for assigned variable.

        We do not handle container values.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(): a = b = 2")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['a']
        Interval(low=2, high=2)
        >>> res['b']
        Interval(low=2, high=2)
        """
        assigned_range = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.add(target.id, assigned_range)
            else:
                self.visit(target)

    def visit_AugAssign(self, node):
        """ Update range value for augassigned variables.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(): a = 2; a -= 1")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['a']
        Interval(low=1, high=1)
        """
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            name = node.target.id
            res = combine(node.op, self.result[name], self.result[node.value])
            self.result[name] = res

    def visit_For(self, node):
        """ Handle iterate variable in for loops.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = b = c = 2
        ...     for i in builtins.range(1):
        ...         a -= 1
        ...         b += 1''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['a']
        Interval(low=-inf, high=2)
        >>> res['b']
        Interval(low=2, high=inf)
        >>> res['c']
        Interval(low=2, high=2)

        >>> node = ast.parse('''
        ... def foo():
        ...     for i in (1, 2, 4):
        ...         a = i''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['a']
        Interval(low=1, high=4)
        """
        assert isinstance(node.target, ast.Name), 'For apply on variables.'
        self.visit(node.iter)
        if isinstance(node.iter, ast.Call):
            for alias in self.aliases[node.iter.func]:
                if isinstance(alias, Intrinsic):
                    self.add(node.target.id, alias.return_range_content([self.visit(n) for n in node.iter.args]))
        self.visit_loop(node, ast.Compare(node.target, [ast.In()], [node.iter]))

    def visit_loop(self, node, cond=None):
        """ Handle incremented variables in loop body.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = b = c = 2
        ...     while a > 0:
        ...         a -= 1
        ...         b += 1''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['a']
        Interval(low=0, high=2)
        >>> res['b']
        Interval(low=2, high=inf)
        >>> res['c']
        Interval(low=2, high=2)
        """
        if cond is not None:
            init_range = self.result
            self.result = self.result.copy()
            bound_range(self.result, self.aliases, cond)
        for stmt in node.body:
            self.visit(stmt)
        old_range = self.result.copy()
        for stmt in node.body:
            self.visit(stmt)
        for expr, range_ in old_range.items():
            self.result[expr] = self.result[expr].widen(range_)
        if cond is not None:
            bound_range(self.result, self.aliases, cond)
            for stmt in node.body:
                self.visit(stmt)
            self.unionify(init_range)
            self.visit(cond)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node):
        self.visit(node.test)
        return self.visit_loop(node, node.test)

    def visit_If(self, node):
        """ Handle iterate variable across branches

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> pm = passmanager.PassManager("test")

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 1: b = 1
        ...     else: b = 3''')

        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 1: b = a
        ...     else: b = 3''')
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['b']
        Interval(low=2, high=inf)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if 0 < a < 4: b = a
        ...     else: b = 3''')
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if (0 < a) and (a < 4): b = a
        ...     else: b = 3''')
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if (a == 1) or (a == 2): b = a
        ...     else: b = 3''')
        >>> res = pm.gather(RangeValuesSimple, node)
        >>> res['b']
        Interval(low=1, high=3)
        """
        self.visit(node.test)
        old_range = self.result
        self.result = old_range.copy()
        bound_range(self.result, self.aliases, node.test)
        for stmt in node.body:
            self.visit(stmt)
        body_range = self.result
        self.result = old_range.copy()
        for stmt in node.orelse:
            self.visit(stmt)
        orelse_range = self.result
        self.result = body_range
        self.unionify(orelse_range)

    def visit_Try(self, node):
        init_range = self.result
        self.result = init_range.copy()
        for stmt in node.body:
            self.visit(stmt)
        self.unionify(init_range)
        init_range = self.result.copy()
        for handler in node.handlers:
            self.result, prev_state = (init_range.copy(), self.result)
            for stmt in handler.body:
                self.visit(stmt)
            self.unionify(prev_state)
        self.result, prev_state = (init_range, self.result)
        for stmt in node.orelse:
            self.visit(stmt)
        self.unionify(prev_state)
        for stmt in node.finalbody:
            self.visit(stmt)