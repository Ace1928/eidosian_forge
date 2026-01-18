from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_string(ps_object):

    def __str__(self):
        return '(%s)' % repr(self.value)[1:-1]