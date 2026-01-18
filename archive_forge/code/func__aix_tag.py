import sys
import sysconfig
def _aix_tag(vrtl, bd):
    _sz = 32 if sys.maxsize == 2 ** 31 - 1 else 64
    _bd = bd if bd != 0 else 9988
    return 'aix-{:1x}{:1d}{:02d}-{:04d}-{}'.format(vrtl[0], vrtl[1], vrtl[2], _bd, _sz)