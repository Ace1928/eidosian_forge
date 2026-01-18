from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def AddMapSetFlag(group, flag_name, long_name, key_type, value_type, key_metavar='KEY', value_metavar='VALUE'):
    return MapSetFlag(flag_name, long_name, key_type, value_type, key_metavar=key_metavar, value_metavar=value_metavar).AddToParser(group)