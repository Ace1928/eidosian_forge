import os
import collections.abc
def _dpkg_wildcard_to_tuple(self, arch):
    try:
        return self._wildcard_cache[arch]
    except KeyError:
        pass
    arch_tuple = arch.split('-', 3)
    if 'any' in arch_tuple:
        while len(arch_tuple) < 4:
            arch_tuple.insert(0, 'any')
        result = QuadTupleDpkgArchitecture(*arch_tuple)
    else:
        result = self._dpkg_arch_to_tuple(arch)
    self._wildcard_cache[arch] = result
    return result