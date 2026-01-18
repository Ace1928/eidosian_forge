import re
import itertools
class QRKanji(QR):
    bits = (13,)
    group = 1
    mode = 8
    lengthbits = (8, 10, 12)

    def __init__(self, data):
        try:
            self.data = self.unicode_to_qrkanji(data)
        except UnicodeEncodeError:
            raise ValueError('Not valid kanji')

    def unicode_to_qrkanji(self, data):
        codes = []
        for i, c in enumerate(data):
            try:
                c = c.encode('shift-jis')
                try:
                    c, d = map(ord, c)
                except TypeError:
                    c, d = c
            except UnicodeEncodeError as e:
                raise UnicodeEncodeError('qrkanji', data, i, i + 1, e.args[4])
            except ValueError:
                raise UnicodeEncodeError('qrkanji', data, i, i + 1, 'illegal multibyte sequence')
            c = c << 8 | d
            if 33088 <= c <= 40956:
                c -= 33088
                c = ((c & 65280) >> 8) * 192 + (c & 255)
            elif 57408 <= c <= 60351:
                c -= 49472
                c = ((c & 65280) >> 8) * 192 + (c & 255)
            else:
                raise UnicodeEncodeError('qrkanji', data, i, i + 1, 'illegal multibyte sequence')
            codes.append(c)
        return codes

    def write(self, buffer, version):
        self.write_header(buffer, version)
        for d in self.data:
            buffer.put(d, 13)