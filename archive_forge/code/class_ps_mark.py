from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_mark(ps_object):

    def __init__(self):
        self.value = 'mark'
        self.type = self.__class__.__name__[3:] + 'type'