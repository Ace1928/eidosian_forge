from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def MapClearFlag(flag_name, long_name):
    return base.Argument('--clear-{}'.format(flag_name), action='store_true', help='Remove all {}.'.format(long_name))