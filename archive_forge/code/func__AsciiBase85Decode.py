import glob
import os
from io import StringIO
def _AsciiBase85Decode(input):
    """This is not used - Acrobat Reader decodes for you - but a round
      trip is essential for testing."""
    outstream = StringIO()
    stripped = ''.join(input.split(), '')
    assert stripped[-2:] == '~>', 'Invalid terminator for Ascii Base 85 Stream'
    stripped = stripped[:-2]
    stripped = stripped.replace('z', '!!!!!')
    whole_word_count, remainder_size = divmod(len(stripped), 5)
    assert remainder_size != 1, 'invalid Ascii 85 stream!'
    cut = 5 * whole_word_count
    body, lastbit = (stripped[0:cut], stripped[cut:])
    for i in range(whole_word_count):
        offset = i * 5
        c1 = ord(body[offset]) - 33
        c2 = ord(body[offset + 1]) - 33
        c3 = ord(body[offset + 2]) - 33
        c4 = ord(body[offset + 3]) - 33
        c5 = ord(body[offset + 4]) - 33
        num = 85 ** 4 * c1 + 85 ** 3 * c2 + 85 ** 2 * c3 + 85 * c4 + c5
        temp, b4 = divmod(num, 256)
        temp, b3 = divmod(temp, 256)
        b1, b2 = divmod(temp, 256)
        assert num == 16777216 * b1 + 65536 * b2 + 256 * b3 + b4, 'dodgy code!'
        outstream.write(chr(b1))
        outstream.write(chr(b2))
        outstream.write(chr(b3))
        outstream.write(chr(b4))
    if remainder_size > 0:
        while len(lastbit) < 5:
            lastbit = lastbit + '!'
        c1 = ord(lastbit[0]) - 33
        c2 = ord(lastbit[1]) - 33
        c3 = ord(lastbit[2]) - 33
        c4 = ord(lastbit[3]) - 33
        c5 = ord(lastbit[4]) - 33
        num = 85 ** 4 * c1 + 85 ** 3 * c2 + 85 ** 2 * c3 + 85 * c4 + c5
        temp, b4 = divmod(num, 256)
        temp, b3 = divmod(temp, 256)
        b1, b2 = divmod(temp, 256)
        assert num == 16777216 * b1 + 65536 * b2 + 256 * b3 + b4, 'dodgy code!'
        if remainder_size == 2:
            lastword = chr(b1 + 1)
        elif remainder_size == 3:
            lastword = chr(b1) + chr(b2 + 1)
        elif remainder_size == 4:
            lastword = chr(b1) + chr(b2) + chr(b3 + 1)
        outstream.write(lastword)
    outstream.reset()
    return outstream.read()