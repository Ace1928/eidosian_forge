from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
import six
def Parameters(value):
    """Declarative command argument type for parameters flag."""
    return arg_parsers.ArgDict(key_type=_FormatExtendedOptions)(value)