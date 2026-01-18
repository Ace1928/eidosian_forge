from .core import FunctionExpression
class ExprFunc:

    def __init__(self, name, doc):
        self.name = name
        self.doc = doc
        self.__doc__ = '{}(*args)\n    {}'.format(name, doc)

    def __call__(self, *args):
        return FunctionExpression(self.name, args)

    def __repr__(self):
        return '<function expr.{}(*args)>'.format(self.name)