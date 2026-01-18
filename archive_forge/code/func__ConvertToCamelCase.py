from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.service_extensions import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
def _ConvertToCamelCase(name):
    """Converts kebab-case name to camelCase."""
    part = name.split('-')
    return part[0] + ''.join((x.title() for x in part[1:]))