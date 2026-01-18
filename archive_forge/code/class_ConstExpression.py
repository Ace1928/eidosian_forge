from ..utils import SchemaBase
class ConstExpression(Expression):

    def __init__(self, name, doc):
        self.__doc__ = '{}: {}'.format(name, doc)
        super(ConstExpression, self).__init__(name=name, doc=doc)

    def __repr__(self):
        return str(self.name)