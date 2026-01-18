from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def MapRemoveFlag(flag_name, long_name, key_type, key_metavar='KEY'):
    return base.Argument('--remove-{}'.format(flag_name), metavar=key_metavar, action=arg_parsers.UpdateAction, type=arg_parsers.ArgList(element_type=key_type), help='List of {} to be removed.'.format(long_name))