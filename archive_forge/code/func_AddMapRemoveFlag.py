from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def AddMapRemoveFlag(group, flag_name, long_name, key_type, key_metavar='KEY'):
    return MapRemoveFlag(flag_name, long_name, key_type, key_metavar=key_metavar).AddToParser(group)