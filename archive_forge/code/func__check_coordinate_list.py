import struct
import dns.exception
import dns.immutable
import dns.rdata
def _check_coordinate_list(value, low, high):
    if value[0] < low or value[0] > high:
        raise ValueError(f'not in range [{low}, {high}]')
    if value[1] < 0 or value[1] > 59:
        raise ValueError('bad minutes value')
    if value[2] < 0 or value[2] > 59:
        raise ValueError('bad seconds value')
    if value[3] < 0 or value[3] > 999:
        raise ValueError('bad milliseconds value')
    if value[4] != 1 and value[4] != -1:
        raise ValueError('bad hemisphere value')