import builtins
import configparser
import operator
import sys
from cherrypy._cpcompat import text_or_bytes
class _Builder:

    def build(self, o):
        m = getattr(self, 'build_' + o.__class__.__name__, None)
        if m is None:
            raise TypeError('unrepr does not recognize %s' % repr(o.__class__.__name__))
        return m(o)

    def astnode(self, s):
        """Return a Python3 ast Node compiled from a string."""
        try:
            import ast
        except ImportError:
            return eval(s)
        p = ast.parse('__tempvalue__ = ' + s)
        return p.body[0].value

    def build_Subscript(self, o):
        return self.build(o.value)[self.build(o.slice)]

    def build_Index(self, o):
        return self.build(o.value)

    def _build_call35(self, o):
        """
        Workaround for python 3.5 _ast.Call signature, docs found here
        https://greentreesnakes.readthedocs.org/en/latest/nodes.html
        """
        import ast
        callee = self.build(o.func)
        args = []
        if o.args is not None:
            for a in o.args:
                if isinstance(a, ast.Starred):
                    args.append(self.build(a.value))
                else:
                    args.append(self.build(a))
        kwargs = {}
        for kw in o.keywords:
            if kw.arg is None:
                rst = self.build(kw.value)
                if not isinstance(rst, dict):
                    raise TypeError('Invalid argument for call.Must be a mapping object.')
                for k, v in rst.items():
                    if k not in kwargs:
                        kwargs[k] = v
            else:
                kwargs[kw.arg] = self.build(kw.value)
        return callee(*args, **kwargs)

    def build_Call(self, o):
        if sys.version_info >= (3, 5):
            return self._build_call35(o)
        callee = self.build(o.func)
        if o.args is None:
            args = ()
        else:
            args = tuple([self.build(a) for a in o.args])
        if o.starargs is None:
            starargs = ()
        else:
            starargs = tuple(self.build(o.starargs))
        if o.kwargs is None:
            kwargs = {}
        else:
            kwargs = self.build(o.kwargs)
        if o.keywords is not None:
            for kw in o.keywords:
                kwargs[kw.arg] = self.build(kw.value)
        return callee(*args + starargs, **kwargs)

    def build_List(self, o):
        return list(map(self.build, o.elts))

    def build_Str(self, o):
        return o.s

    def build_Num(self, o):
        return o.n

    def build_Dict(self, o):
        return dict([(self.build(k), self.build(v)) for k, v in zip(o.keys, o.values)])

    def build_Tuple(self, o):
        return tuple(self.build_List(o))

    def build_Name(self, o):
        name = o.id
        if name == 'None':
            return None
        if name == 'True':
            return True
        if name == 'False':
            return False
        try:
            return modules(name)
        except ImportError:
            pass
        try:
            return getattr(builtins, name)
        except AttributeError:
            pass
        raise TypeError('unrepr could not resolve the name %s' % repr(name))

    def build_NameConstant(self, o):
        return o.value
    build_Constant = build_NameConstant

    def build_UnaryOp(self, o):
        op, operand = map(self.build, [o.op, o.operand])
        return op(operand)

    def build_BinOp(self, o):
        left, op, right = map(self.build, [o.left, o.op, o.right])
        return op(left, right)

    def build_Add(self, o):
        return operator.add

    def build_Mult(self, o):
        return operator.mul

    def build_USub(self, o):
        return operator.neg

    def build_Attribute(self, o):
        parent = self.build(o.value)
        return getattr(parent, o.attr)

    def build_NoneType(self, o):
        return None