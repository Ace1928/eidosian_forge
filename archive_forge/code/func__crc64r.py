import sys
def _crc64r(data, crc, table):
    crc = crc & long(18446744073709551615)
    for x in data:
        crc = table[ord(x) ^ int(crc & long(255))] ^ crc >> 8
    return crc