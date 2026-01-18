from collections import namedtuple
def _print_registers(self, vfp_mask, prefix):
    hits = [prefix + str(i) for i in range(32) if vfp_mask & 1 << i != 0]
    return '{%s}' % ', '.join(hits)