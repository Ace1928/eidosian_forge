from ..utils import SchemaBase
class FunctionExpression(Expression):

    def __init__(self, name, args):
        super(FunctionExpression, self).__init__(name=name, args=args)

    def __repr__(self):
        args = ','.join((_js_repr(arg) for arg in self.args))
        return '{name}({args})'.format(name=self.name, args=args)