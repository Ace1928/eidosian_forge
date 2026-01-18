import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _XCKVPrint(self, file, tabs, key, value):
    """Prints a key and value, members of an XCObject's _properties dictionary,
    to file.

    tabs is an int identifying the indentation level.  If the class'
    _should_print_single_line variable is True, tabs is ignored and the
    key-value pair will be followed by a space insead of a newline.
    """
    if self._should_print_single_line:
        printable = ''
        after_kv = ' '
    else:
        printable = '\t' * tabs
        after_kv = '\n'
    if key == 'remoteGlobalIDString' and isinstance(self, PBXContainerItemProxy):
        value_to_print = value.id
    else:
        value_to_print = value
    if key == 'settings' and isinstance(self, PBXBuildFile):
        strip_value_quotes = True
    else:
        strip_value_quotes = False
    if key == 'buildSettings' and isinstance(self, XCBuildConfiguration):
        flatten_list = True
    else:
        flatten_list = False
    try:
        printable_key = self._XCPrintableValue(tabs, key, flatten_list)
        printable_value = self._XCPrintableValue(tabs, value_to_print, flatten_list)
        if strip_value_quotes and len(printable_value) > 1 and (printable_value[0] == '"') and (printable_value[-1] == '"'):
            printable_value = printable_value[1:-1]
        printable += printable_key + ' = ' + printable_value + ';' + after_kv
    except TypeError as e:
        gyp.common.ExceptionAppend(e, 'while printing key "%s"' % key)
        raise
    self._XCPrint(file, 0, printable)