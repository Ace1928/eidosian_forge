from ..utils import SchemaBase
class BinaryExpression(Expression):

    def __init__(self, op, lhs, rhs):
        super(BinaryExpression, self).__init__(op=op, lhs=lhs, rhs=rhs)

    def __repr__(self):
        return '({lhs} {op} {rhs})'.format(op=self.op, lhs=_js_repr(self.lhs), rhs=_js_repr(self.rhs))