from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_operator(ps_object):
    literal = 0

    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.type = self.__class__.__name__[3:] + 'type'

    def __repr__(self):
        return '<operator %s>' % self.name