class TfmReader(object):

    def __init__(self, f):
        self.f = f

    def read_byte(self):
        return ord(self.f.read(1))

    def read_halfword(self):
        b1 = self.read_byte()
        b2 = self.read_byte()
        return b1 << 8 | b2

    def read_word(self):
        b1 = self.read_byte()
        b2 = self.read_byte()
        b3 = self.read_byte()
        b4 = self.read_byte()
        return b1 << 24 | b2 << 16 | b3 << 8 | b4

    def read_fixword(self):
        word = self.read_word()
        neg = False
        if word & 2147483648:
            neg = True
            word = -word & 4294967295
        return (-1 if neg else 1) * word / float(1 << 20)

    def read_bcpl(self, length):
        str_length = self.read_byte()
        data = self.f.read(length - 1)
        return data[:str_length]