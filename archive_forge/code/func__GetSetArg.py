from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from six.moves import map  # pylint: disable=redefined-builtin
def _GetSetArg(arg_name, metavar, prop_name, is_dict_args):
    list_help = 'Completely replace the current {} with the given values.'.format(prop_name)
    dict_help = 'Completely replace the current {} with the given key-value pairs.'.format(prop_name)
    return base.Argument('--set-{}'.format(arg_name), type=_GetArgType(is_dict_args), metavar=metavar, help=_GetArgHelp(dict_help, list_help, is_dict_args))