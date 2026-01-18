from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ruamel import yaml
import six
def _ScalarType(self, node):
    if isinstance(node.value, six.string_types):
        if node.style == '|':
            return _LvPreservedScalarString(node.value)
        if self._preserve_quotes:
            if node.style == "'":
                return _LvSingleQuotedScalarString(node.value)
            if node.style == '"':
                return _LvDoubleQuotedScalarString(node.value)
    return _LvString(node.value)