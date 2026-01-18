import re
import itertools
class QRHanzi(QR):
    bits = (13,)
    group = 1
    mode = 13
    lengthbits = (8, 10, 12)

    def __init__(self, data):
        try:
            self.data = self.unicode_to_qrhanzi(data)
        except UnicodeEncodeError:
            raise ValueError('Not valid hanzi')

    def unicode_to_qrhanzi(self, data):
        codes = []
        for i, c in enumerate(data):
            try:
                c = c.encode('gb2312')
                try:
                    c, d = map(ord, c)
                except TypeError:
                    c, d = c
            except UnicodeEncodeError as e:
                raise UnicodeEncodeError('qrhanzi', data, i, i + 1, e.args[4])
            except ValueError:
                raise UnicodeEncodeError('qrhanzi', data, i, i + 1, 'illegal multibyte sequence')
            c = c << 8 | d
            if 41377 <= c <= 43774:
                c -= 41377
                c = ((c & 65280) >> 8) * 96 + (c & 255)
            elif 45217 <= c <= 64254:
                c -= 42657
                c = ((c & 65280) >> 8) * 96 + (c & 255)
            else:
                raise UnicodeEncodeError('qrhanzi', data, i, i + 1, 'illegal multibyte sequence')
            codes.append(c)
        return codes

    def write_header(self, buffer, version):
        buffer.put(self.mode, 4)
        buffer.put(1, 4)
        lenbits = self.getLengthBits(version)
        if lenbits:
            buffer.put(len(self.data), lenbits)

    def write(self, buffer, version):
        self.write_header(buffer, version)
        for d in self.data:
            buffer.put(d, 13)