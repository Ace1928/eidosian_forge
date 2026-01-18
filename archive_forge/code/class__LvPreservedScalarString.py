from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ruamel import yaml
import six
class _LvPreservedScalarString(yaml.scalarstring.PreservedScalarString):
    """Location/value preserved scalar string type."""
    __slots__ = ('lc', 'value')