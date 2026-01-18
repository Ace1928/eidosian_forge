from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ruamel import yaml
import six
class _LvSingleQuotedScalarString(yaml.scalarstring.SingleQuotedScalarString):
    """Location/value single quoted scalar string type."""
    __slots__ = ('lc', 'value')