from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
import six
def _FormatExtendedOptions(key):
    """Replaces dash with underscore for extended options parameters."""
    if key in MEMCACHE_EXTENDED_OPTIONS:
        return key.replace('-', '_')
    return key