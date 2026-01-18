import sys
def _crc8r(data, crc, table):
    crc = crc & 255
    for x in data:
        crc = table[ord(x) ^ crc]
    return crc