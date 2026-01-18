import os
import collections.abc
class QuadTupleDpkgArchitecture(_QuadTuple):
    """Implementation detail of ArchTable"""

    def __contains__(self, item):
        if isinstance(item, QuadTupleDpkgArchitecture):
            return self.api_name in ('any', item.api_name) and self.libc_name in ('any', item.libc_name) and (self.os_name in ('any', item.os_name)) and (self.cpu_name in ('any', item.cpu_name))
        return super().__contains__(item)

    @property
    def is_wildcard(self):
        return any((x == 'any' for x in self))