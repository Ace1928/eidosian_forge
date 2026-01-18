from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_name(ps_object):
    literal = 0

    def __str__(self):
        if self.literal:
            return '/' + self.value
        else:
            return self.value