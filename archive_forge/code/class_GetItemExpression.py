from ..utils import SchemaBase
class GetItemExpression(Expression):

    def __init__(self, group, name):
        super(GetItemExpression, self).__init__(group=group, name=name)

    def __repr__(self):
        return '{}[{!r}]'.format(self.group, self.name)