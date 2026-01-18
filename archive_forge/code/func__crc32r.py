import sys
def _crc32r(data, crc, table):
    crc = crc & long(4294967295)
    for x in data:
        crc = table[ord(x) ^ int(crc & long(255))] ^ crc >> 8
    return crc