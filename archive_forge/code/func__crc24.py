import sys
def _crc24(data, crc, table):
    crc = crc & 16777215
    for x in data:
        crc = table[ord(x) ^ int(crc >> 16) & 255] ^ crc << 8 & 16776960
    return crc