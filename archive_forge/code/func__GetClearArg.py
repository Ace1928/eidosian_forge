from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from six.moves import map  # pylint: disable=redefined-builtin
def _GetClearArg(arg_name, prop_name):
    return base.Argument('--clear-{}'.format(arg_name), action='store_true', help='Empty the current {}.'.format(prop_name))