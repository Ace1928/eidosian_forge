import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class MemoryBasicInformation(object):
    """
    Memory information object returned by L{VirtualQueryEx}.
    """
    READABLE = PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY | PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY
    WRITEABLE = PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY | PAGE_READWRITE | PAGE_WRITECOPY
    COPY_ON_WRITE = PAGE_EXECUTE_WRITECOPY | PAGE_WRITECOPY
    EXECUTABLE = PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY
    EXECUTABLE_AND_WRITEABLE = PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY

    def __init__(self, mbi=None):
        """
        @type  mbi: L{MEMORY_BASIC_INFORMATION} or L{MemoryBasicInformation}
        @param mbi: Either a L{MEMORY_BASIC_INFORMATION} structure or another
            L{MemoryBasicInformation} instance.
        """
        if mbi is None:
            self.BaseAddress = None
            self.AllocationBase = None
            self.AllocationProtect = None
            self.RegionSize = None
            self.State = None
            self.Protect = None
            self.Type = None
        else:
            self.BaseAddress = mbi.BaseAddress
            self.AllocationBase = mbi.AllocationBase
            self.AllocationProtect = mbi.AllocationProtect
            self.RegionSize = mbi.RegionSize
            self.State = mbi.State
            self.Protect = mbi.Protect
            self.Type = mbi.Type
            if hasattr(mbi, 'content'):
                self.content = mbi.content
            if hasattr(mbi, 'filename'):
                self.content = mbi.filename

    def __contains__(self, address):
        """
        Test if the given memory address falls within this memory region.

        @type  address: int
        @param address: Memory address to test.

        @rtype:  bool
        @return: C{True} if the given memory address falls within this memory
            region, C{False} otherwise.
        """
        return self.BaseAddress <= address < self.BaseAddress + self.RegionSize

    def is_free(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region is free.
        """
        return self.State == MEM_FREE

    def is_reserved(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region is reserved.
        """
        return self.State == MEM_RESERVE

    def is_commited(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region is commited.
        """
        return self.State == MEM_COMMIT

    def is_image(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region belongs to an executable
            image.
        """
        return self.Type == MEM_IMAGE

    def is_mapped(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region belongs to a mapped file.
        """
        return self.Type == MEM_MAPPED

    def is_private(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region is private.
        """
        return self.Type == MEM_PRIVATE

    def is_guard(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are guard pages.
        """
        return self.is_commited() and bool(self.Protect & PAGE_GUARD)

    def has_content(self):
        """
        @rtype:  bool
        @return: C{True} if the memory in this region has any data in it.
        """
        return self.is_commited() and (not bool(self.Protect & (PAGE_GUARD | PAGE_NOACCESS)))

    def is_readable(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are readable.
        """
        return self.has_content() and bool(self.Protect & self.READABLE)

    def is_writeable(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are writeable.
        """
        return self.has_content() and bool(self.Protect & self.WRITEABLE)

    def is_copy_on_write(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are marked as
            copy-on-write. This means the pages are writeable, but changes
            are not propagated to disk.
        @note:
            Tipically data sections in executable images are marked like this.
        """
        return self.has_content() and bool(self.Protect & self.COPY_ON_WRITE)

    def is_executable(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are executable.
        @note: Executable pages are always readable.
        """
        return self.has_content() and bool(self.Protect & self.EXECUTABLE)

    def is_executable_and_writeable(self):
        """
        @rtype:  bool
        @return: C{True} if all pages in this region are executable and
            writeable.
        @note: The presence of such pages make memory corruption
            vulnerabilities much easier to exploit.
        """
        return self.has_content() and bool(self.Protect & self.EXECUTABLE_AND_WRITEABLE)