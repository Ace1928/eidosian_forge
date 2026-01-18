from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def _ExtensionDebugString(self, prefix, printElemNumber):
    res = ''
    extensions = self._ListExtensions()
    for extension in extensions:
        value = self._extension_fields[extension]
        if extension.is_repeated:
            cnt = 0
            for e in value:
                elm = ''
                if printElemNumber:
                    elm = '(%d)' % cnt
                if extension.composite_cls is not None:
                    res += prefix + '[%s%s] {\n' % (extension.full_name, elm)
                    res += e.__str__(prefix + '  ', printElemNumber)
                    res += prefix + '}\n'
        elif extension.composite_cls is not None:
            res += prefix + '[%s] {\n' % extension.full_name
            res += value.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
        else:
            if extension.field_type in _TYPE_TO_DEBUG_STRING:
                text_value = _TYPE_TO_DEBUG_STRING[extension.field_type](self, value)
            else:
                text_value = self.DebugFormat(value)
            res += prefix + '[%s]: %s\n' % (extension.full_name, text_value)
    return res