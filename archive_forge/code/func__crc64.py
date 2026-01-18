import sys
def _crc64(data, crc, table):
    crc = crc & long(18446744073709551615)
    for x in data:
        crc = table[ord(x) ^ int(crc >> 56) & 255] ^ crc << 8 & long(18446744073709551360)
    return crc