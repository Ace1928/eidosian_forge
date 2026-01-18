from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_literal(ps_object):

    def __str__(self):
        return '/' + self.value