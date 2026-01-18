from collections import namedtuple
def _calculate_range(self, start, count):
    return (1 << count + 1) - 1 << start