from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from six.moves import map  # pylint: disable=redefined-builtin
def AddPrimitiveArgs(parser, resource_name, arg_name, property_name, additional_help='', metavar=None, is_dict_args=False, auto_group_help=True, include_set=True):
    """Add arguments for updating a field to the given parser.

  Adds `--{add,remove,set,clear-<resource>` arguments.

  Args:
    parser: calliope.parser_extensions.ArgumentInterceptor, the parser to add
      arguments to.
    resource_name: str, the (singular) name of the resource being modified (in
      whatever format you'd like it to appear in help text).
    arg_name: str, the (plural) argument suffix to use (hyphen-case).
    property_name: str, the description of the property being modified (plural;
      in whatever format you'd like it to appear in help text)
    additional_help: str, additional help text describing the property.
    metavar: str, the name of the metavar to use (if different from
      arg_name.upper()).
    is_dict_args: boolean, True when the primitive args are dict args.
    auto_group_help: bool, True to generate a summary help.
    include_set: bool, True to include the (deprecated) set argument.
  """
    properties_name = property_name
    if auto_group_help:
        group_help = 'These flags modify the member {} of this {}.'.format(properties_name, resource_name)
        if additional_help:
            group_help += ' ' + additional_help
    else:
        group_help = additional_help
    group = parser.add_mutually_exclusive_group(group_help)
    metavar = metavar or arg_name.upper()
    args = [_GetAppendArg(arg_name, metavar, properties_name, is_dict_args), _GetRemoveArg(arg_name, metavar, properties_name, is_dict_args), _GetClearArg(arg_name, properties_name)]
    if include_set:
        args.append(_GetSetArg(arg_name, metavar, properties_name, is_dict_args))
    for arg in args:
        arg.AddToParser(group)