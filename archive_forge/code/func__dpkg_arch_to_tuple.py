import os
import collections.abc
def _dpkg_arch_to_tuple(self, dpkg_arch):
    if dpkg_arch.startswith('linux-'):
        dpkg_arch = dpkg_arch[6:]
    return self._arch2table[dpkg_arch]