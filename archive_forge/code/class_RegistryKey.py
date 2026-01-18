from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
import collections
import warnings
class RegistryKey(_RegistryContainer):
    """
    Exposes a single Windows Registry key as a dictionary-like object.

    @see: L{Registry}

    @type path: str
    @ivar path: Registry key path.

    @type handle: L{win32.RegistryKeyHandle}
    @ivar handle: Registry key handle.
    """

    def __init__(self, path, handle):
        """
        @type  path: str
        @param path: Registry key path.

        @type  handle: L{win32.RegistryKeyHandle}
        @param handle: Registry key handle.
        """
        super(RegistryKey, self).__init__()
        if path.endswith('\\'):
            path = path[:-1]
        self._path = path
        self._handle = handle

    @property
    def path(self):
        return self._path

    @property
    def handle(self):
        return self._handle

    def __contains__(self, name):
        try:
            win32.RegQueryValueEx(self.handle, name, False)
            return True
        except WindowsError:
            e = sys.exc_info()[1]
            if e.winerror == win32.ERROR_FILE_NOT_FOUND:
                return False
            raise

    def __getitem__(self, name):
        try:
            return win32.RegQueryValueEx(self.handle, name)[0]
        except WindowsError:
            e = sys.exc_info()[1]
            if e.winerror == win32.ERROR_FILE_NOT_FOUND:
                raise KeyError(name)
            raise

    def __setitem__(self, name, value):
        win32.RegSetValueEx(self.handle, name, value)

    def __delitem__(self, name):
        win32.RegDeleteValue(self.handle, name)

    def iterkeys(self):
        handle = self.handle
        index = 0
        while 1:
            resp = win32.RegEnumValue(handle, index, False)
            if resp is None:
                break
            yield resp[0]
            index += 1

    def itervalues(self):
        handle = self.handle
        index = 0
        while 1:
            resp = win32.RegEnumValue(handle, index)
            if resp is None:
                break
            yield resp[2]
            index += 1

    def iteritems(self):
        handle = self.handle
        index = 0
        while 1:
            resp = win32.RegEnumValue(handle, index)
            if resp is None:
                break
            yield (resp[0], resp[2])
            index += 1

    def keys(self):
        handle = self.handle
        keys = list()
        index = 0
        while 1:
            resp = win32.RegEnumValue(handle, index, False)
            if resp is None:
                break
            keys.append(resp[0])
            index += 1
        return keys

    def values(self):
        handle = self.handle
        values = list()
        index = 0
        while 1:
            resp = win32.RegEnumValue(handle, index)
            if resp is None:
                break
            values.append(resp[2])
            index += 1
        return values

    def items(self):
        handle = self.handle
        items = list()
        index = 0
        while 1:
            resp = win32.RegEnumValue(handle, index)
            if resp is None:
                break
            items.append((resp[0], resp[2]))
            index += 1
        return items

    def get_value_type(self, name):
        """
        Retrieves the low-level data type for the given value.

        @type  name: str
        @param name: Registry value name.

        @rtype:  int
        @return: One of the following constants:
         - L{win32.REG_NONE} (0)
         - L{win32.REG_SZ} (1)
         - L{win32.REG_EXPAND_SZ} (2)
         - L{win32.REG_BINARY} (3)
         - L{win32.REG_DWORD} (4)
         - L{win32.REG_DWORD_BIG_ENDIAN} (5)
         - L{win32.REG_LINK} (6)
         - L{win32.REG_MULTI_SZ} (7)
         - L{win32.REG_RESOURCE_LIST} (8)
         - L{win32.REG_FULL_RESOURCE_DESCRIPTOR} (9)
         - L{win32.REG_RESOURCE_REQUIREMENTS_LIST} (10)
         - L{win32.REG_QWORD} (11)

        @raise KeyError: The specified value could not be found.
        """
        try:
            return win32.RegQueryValueEx(self.handle, name)[1]
        except WindowsError:
            e = sys.exc_info()[1]
            if e.winerror == win32.ERROR_FILE_NOT_FOUND:
                raise KeyError(name)
            raise

    def clear(self):
        handle = self.handle
        while 1:
            resp = win32.RegEnumValue(handle, 0, False)
            if resp is None:
                break
            win32.RegDeleteValue(handle, resp[0])

    def __str__(self):
        default = self['']
        return str(default)

    def __unicode__(self):
        default = self[u'']
        return compat.unicode(default)

    def __repr__(self):
        return '<Registry key: "%s">' % self._path

    def iterchildren(self):
        """
        Iterates the subkeys for this Registry key.

        @rtype:  iter of L{RegistryKey}
        @return: Iterator of subkeys.
        """
        handle = self.handle
        index = 0
        while 1:
            subkey = win32.RegEnumKey(handle, index)
            if subkey is None:
                break
            yield self.child(subkey)
            index += 1

    def children(self):
        """
        Returns a list of subkeys for this Registry key.

        @rtype:  list(L{RegistryKey})
        @return: List of subkeys.
        """
        handle = self.handle
        result = []
        index = 0
        while 1:
            subkey = win32.RegEnumKey(handle, index)
            if subkey is None:
                break
            result.append(self.child(subkey))
            index += 1
        return result

    def child(self, subkey):
        """
        Retrieves a subkey for this Registry key, given its name.

        @type  subkey: str
        @param subkey: Name of the subkey.

        @rtype:  L{RegistryKey}
        @return: Subkey.
        """
        path = self._path + '\\' + subkey
        handle = win32.RegOpenKey(self.handle, subkey)
        return RegistryKey(path, handle)

    def flush(self):
        """
        Flushes changes immediately to disk.

        This method is normally not needed, as the Registry writes changes
        to disk by itself. This mechanism is provided to ensure the write
        happens immediately, as opposed to whenever the OS wants to.

        @warn: Calling this method too often may degrade performance.
        """
        win32.RegFlushKey(self.handle)