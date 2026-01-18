import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
class RangeValuesBase(ModuleAnalysis):
    ResultHolder = object()

    def __init__(self):
        """Initialize instance variable and gather globals name information."""
        self.result = defaultdict(lambda: UNKNOWN_RANGE)
        from pythran.analyses import UseOMP
        super(RangeValuesBase, self).__init__(Aliases, CFG, UseOMP)
        self.parent = self

    def add(self, variable, range_):
        """
        Add a new low and high bound for a variable.

        As it is flow insensitive, it compares it with old values and update it
        if needed.
        """
        if variable not in self.result:
            self.result[variable] = range_
        else:
            self.result[variable] = self.result[variable].union(range_)
        return self.result[variable]

    def unionify(self, other):
        for k, v in other.items():
            if k in self.result:
                self.result[k] = self.result[k].union(v)
            else:
                self.result[k] = v

    def widen(self, curr, other):
        self.result = curr
        for k, v in other.items():
            w = self.result.get(k, None)
            if w is None:
                self.result[k] = v
            elif v is not w:
                self.result[k] = w.widen(v)

    def visit_BoolOp(self, node):
        """ Merge right and left operands ranges.

        TODO : We could exclude some operand with this range information...

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2
        ...     c = 3
        ...     d = a or c''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['d']
        Interval(low=2, high=3)
        """
        res = list(zip(*[self.visit(elt).bounds() for elt in node.values]))
        return self.add(node, Interval(min(res[0]), max(res[1])))

    def visit_BinOp(self, node):
        """ Combine operands ranges for given operator.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2
        ...     c = 3
        ...     d = a - c''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['d']
        Interval(low=-1, high=-1)
        """
        res = combine(node.op, self.visit(node.left), self.visit(node.right))
        return self.add(node, res)

    def visit_UnaryOp(self, node):
        """ Update range with given unary operation.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2
        ...     c = -a
        ...     d = ~a
        ...     f = +a
        ...     e = not a''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['f']
        Interval(low=2, high=2)
        >>> res['c']
        Interval(low=-2, high=-2)
        >>> res['d']
        Interval(low=-3, high=-3)
        >>> res['e']
        Interval(low=0, high=1)
        """
        res = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            res = Interval(0, 1)
        elif isinstance(node.op, ast.Invert) and isinstance(res.high, int) and isinstance(res.low, int):
            res = Interval(~res.high, ~res.low)
        elif isinstance(node.op, ast.UAdd):
            pass
        elif isinstance(node.op, ast.USub):
            res = Interval(-res.high, -res.low)
        else:
            res = UNKNOWN_RANGE
        return self.add(node, res)

    def visit_IfExp(self, node):
        """ Use worst case for both possible values.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2 or 3
        ...     b = 4 or 5
        ...     c = a if a else b''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['c']
        Interval(low=2, high=5)
        """
        self.visit(node.test)
        body_res = self.visit(node.body)
        orelse_res = self.visit(node.orelse)
        return self.add(node, orelse_res.union(body_res))

    def visit_Compare(self, node):
        """ Boolean are possible index.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = 2 or 3
        ...     b = 4 or 5
        ...     c = a < b
        ...     d = b < 3
        ...     e = b == 4''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['c']
        Interval(low=1, high=1)
        >>> res['d']
        Interval(low=0, high=0)
        >>> res['e']
        Interval(low=0, high=1)
        """
        if any((isinstance(op, (ast.In, ast.NotIn, ast.Is, ast.IsNot)) for op in node.ops)):
            self.generic_visit(node)
            return self.add(node, Interval(0, 1))
        curr = self.visit(node.left)
        res = []
        for op, comparator in zip(node.ops, node.comparators):
            comparator = self.visit(comparator)
            fake = ast.Compare(ast.Name('x', ast.Load(), None, None), [op], [ast.Name('y', ast.Load(), None, None)])
            fake = ast.Expression(fake)
            ast.fix_missing_locations(fake)
            expr = compile(ast.gast_to_ast(fake), '<range_values>', 'eval')
            res.append(eval(expr, {'x': curr, 'y': comparator}))
        if all(res):
            return self.add(node, Interval(1, 1))
        elif any((r.low == r.high == 0 for r in res)):
            return self.add(node, Interval(0, 0))
        else:
            return self.add(node, Interval(0, 1))

    def visit_Call(self, node):
        """ Function calls are not handled for now.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = builtins.range(10)''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=-inf, high=inf)
        """
        for alias in self.aliases[node.func]:
            if alias is MODULES['builtins']['getattr']:
                attr_name = node.args[-1].value
                attribute = attributes[attr_name][-1]
                self.add(node, attribute.return_range(None))
            elif isinstance(alias, Intrinsic):
                alias_range = alias.return_range([self.visit(n) for n in node.args])
                self.add(node, alias_range)
            elif isinstance(alias, ast.FunctionDef):
                if alias not in self.result:
                    state = self.save_state()
                    self.parent.visit(alias)
                    return_range = self.result[alias]
                    self.restore_state(state)
                else:
                    return_range = self.result[alias]
                self.add(node, return_range)
            else:
                self.result.pop(node, None)
                return self.generic_visit(node)
        return self.result[node]

    def visit_Constant(self, node):
        """ Handle literals integers values. """
        if isinstance(node.value, (bool, int)):
            return self.add(node, Interval(node.value, node.value))
        return UNKNOWN_RANGE

    def visit_Name(self, node):
        """ Get range for parameters for examples or false branching. """
        return self.add(node, self.result[node.id])

    def visit_Tuple(self, node):
        return self.add(node, IntervalTuple((self.visit(elt) for elt in node.elts)))

    def visit_Index(self, node):
        return self.add(node, self.visit(node.value))

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Call):
            for alias in self.aliases[node.value.func]:
                if alias is MODULES['builtins']['getattr']:
                    attr_name = node.value.args[-1].value
                    attribute = attributes[attr_name][-1]
                    self.add(node, attribute.return_range_content(None))
                elif isinstance(alias, Intrinsic):
                    self.add(node, alias.return_range_content([self.visit(n) for n in node.value.args]))
                else:
                    return self.generic_visit(node)
            if not self.aliases[node.value.func]:
                return self.generic_visit(node)
            self.visit(node.slice)
            return self.result[node]
        else:
            value = self.visit(node.value)
            slice = self.visit(node.slice)
            return self.add(node, value[slice])

    def visit_FunctionDef(self, node):
        """ Set default range value for globals and attributes.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(a, b): pass")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=-inf, high=inf)
        """
        if node in self.result:
            return
        if self.use_omp:
            return
        self.result[node] = UNKNOWN_RANGE
        prev_result = self.result.get(RangeValuesBase.ResultHolder, None)
        self.function_visitor(node)
        del self.result[node]
        self.add(node, self.result[RangeValuesBase.ResultHolder])
        if prev_result is not None:
            self.result[RangeValuesBase.ResultHolder] = prev_result
        else:
            del self.result[RangeValuesBase.ResultHolder]