from .util import (
import sys
from functools import reduce
class StaticWindowMapManager:
    """Provides a manager which will produce single size cursors that are allowed
    to always map the whole file.

    Clients must be written to specifically know that they are accessing their data
    through a StaticWindowMapManager, as they otherwise have to deal with their window size.

    These clients would have to use a SlidingWindowMapBuffer to hide this fact.

    This type will always use a maximum window size, and optimize certain methods to
    accommodate this fact"""
    __slots__ = ['_fdict', '_window_size', '_max_memory_size', '_max_handle_count', '_memory_size', '_handle_count']
    MapRegionListCls = MapRegionList
    MapWindowCls = MapWindow
    MapRegionCls = MapRegion
    WindowCursorCls = WindowCursor
    _MB_in_bytes = 1024 * 1024

    def __init__(self, window_size=0, max_memory_size=0, max_open_handles=sys.maxsize):
        """initialize the manager with the given parameters.
        :param window_size: if -1, a default window size will be chosen depending on
            the operating system's architecture. It will internally be quantified to a multiple of the page size
            If 0, the window may have any size, which basically results in mapping the whole file at one
        :param max_memory_size: maximum amount of memory we may map at once before releasing mapped regions.
            If 0, a viable default will be set depending on the system's architecture.
            It is a soft limit that is tried to be kept, but nothing bad happens if we have to over-allocate
        :param max_open_handles: if not maxint, limit the amount of open file handles to the given number.
            Otherwise the amount is only limited by the system itself. If a system or soft limit is hit,
            the manager will free as many handles as possible"""
        self._fdict = dict()
        self._window_size = window_size
        self._max_memory_size = max_memory_size
        self._max_handle_count = max_open_handles
        self._memory_size = 0
        self._handle_count = 0
        if window_size < 0:
            coeff = 64
            if is_64_bit():
                coeff = 1024
            self._window_size = coeff * self._MB_in_bytes
        if max_memory_size == 0:
            coeff = 1024
            if is_64_bit():
                coeff = 8192
            self._max_memory_size = coeff * self._MB_in_bytes

    def _collect_lru_region(self, size):
        """Unmap the region which was least-recently used and has no client
        :param size: size of the region we want to map next (assuming its not already mapped partially or full
            if 0, we try to free any available region
        :return: Amount of freed regions

        .. Note::
            We don't raise exceptions anymore, in order to keep the system working, allowing temporary overallocation.
            If the system runs out of memory, it will tell.

        .. TODO::
            implement a case where all unusued regions are discarded efficiently.
            Currently its only brute force
        """
        num_found = 0
        while size == 0 or self._memory_size + size > self._max_memory_size:
            lru_region = None
            lru_list = None
            for regions in self._fdict.values():
                for region in regions:
                    if region.client_count() == 1 and (lru_region is None or region._uc < lru_region._uc):
                        lru_region = region
                        lru_list = regions
            if lru_region is None:
                break
            num_found += 1
            del lru_list[lru_list.index(lru_region)]
            lru_region.increment_client_count(-1)
            self._memory_size -= lru_region.size()
            self._handle_count -= 1
        return num_found

    def _obtain_region(self, a, offset, size, flags, is_recursive):
        """Utility to create a new region - for more information on the parameters,
        see MapCursor.use_region.
        :param a: A regions (a)rray
        :return: The newly created region"""
        if self._memory_size + size > self._max_memory_size:
            self._collect_lru_region(size)
        r = None
        if a:
            assert len(a) == 1
            r = a[0]
        else:
            try:
                r = self.MapRegionCls(a.path_or_fd(), 0, sys.maxsize, flags)
            except Exception:
                if is_recursive:
                    raise
                self._collect_lru_region(0)
                return self._obtain_region(a, offset, size, flags, True)
            self._handle_count += 1
            self._memory_size += r.size()
            a.append(r)
        assert r.includes_ofs(offset)
        return r

    def make_cursor(self, path_or_fd):
        """
        :return: a cursor pointing to the given path or file descriptor.
            It can be used to map new regions of the file into memory

        **Note:** if a file descriptor is given, it is assumed to be open and valid,
        but may be closed afterwards. To refer to the same file, you may reuse
        your existing file descriptor, but keep in mind that new windows can only
        be mapped as long as it stays valid. This is why the using actual file paths
        are preferred unless you plan to keep the file descriptor open.

        **Note:** file descriptors are problematic as they are not necessarily unique, as two
        different files opened and closed in succession might have the same file descriptor id.

        **Note:** Using file descriptors directly is faster once new windows are mapped as it
        prevents the file to be opened again just for the purpose of mapping it."""
        regions = self._fdict.get(path_or_fd)
        if regions is None:
            regions = self.MapRegionListCls(path_or_fd)
            self._fdict[path_or_fd] = regions
        return self.WindowCursorCls(self, regions)

    def collect(self):
        """Collect all available free-to-collect mapped regions
        :return: Amount of freed handles"""
        return self._collect_lru_region(0)

    def num_file_handles(self):
        """:return: amount of file handles in use. Each mapped region uses one file handle"""
        return self._handle_count

    def num_open_files(self):
        """Amount of opened files in the system"""
        return reduce(lambda x, y: x + y, (1 for rlist in self._fdict.values() if len(rlist) > 0), 0)

    def window_size(self):
        """:return: size of each window when allocating new regions"""
        return self._window_size

    def mapped_memory_size(self):
        """:return: amount of bytes currently mapped in total"""
        return self._memory_size

    def max_file_handles(self):
        """:return: maximum amount of handles we may have opened"""
        return self._max_handle_count

    def max_mapped_memory_size(self):
        """:return: maximum amount of memory we may allocate"""
        return self._max_memory_size

    def force_map_handle_removal_win(self, base_path):
        """ONLY AVAILABLE ON WINDOWS
        On windows removing files is not allowed if anybody still has it opened.
        If this process is ourselves, and if the whole process uses this memory
        manager (as far as the parent framework is concerned) we can enforce
        closing all memory maps whose path matches the given base path to
        allow the respective operation after all.
        The respective system must NOT access the closed memory regions anymore !
        This really may only be used if you know that the items which keep
        the cursors alive will not be using it anymore. They need to be recreated !
        :return: Amount of closed handles

        **Note:** does nothing on non-windows platforms"""
        if sys.platform != 'win32':
            return
        num_closed = 0
        for path, rlist in self._fdict.items():
            if path.startswith(base_path):
                for region in rlist:
                    region.release()
                    num_closed += 1
        return num_closed