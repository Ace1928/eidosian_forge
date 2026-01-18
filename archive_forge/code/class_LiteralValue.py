from io import StringIO
class LiteralValue(str):

    def value(self, elem):
        return self