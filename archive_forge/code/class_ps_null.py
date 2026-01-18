from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_null(ps_object):

    def __init__(self):
        self.type = self.__class__.__name__[3:] + 'type'