from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_integer(ps_object):

    def __str__(self):
        return repr(self.value)