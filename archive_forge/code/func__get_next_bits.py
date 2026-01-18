from reportlab.graphics.barcode.common import Barcode
def _get_next_bits(self, data):
    value = data.pop(0)
    bits = []
    for i in range(0, 8):
        bits.append(value >> i & 1)
    bits.reverse()
    return bits