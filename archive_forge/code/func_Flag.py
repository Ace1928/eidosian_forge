from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
import six
def Flag(d):
    """Returns a flag object suitable for the calliope.markdown module."""
    flag = type(FLAG_TYPE_NAME, (object,), d)
    flag.is_group = False
    flag.is_hidden = d.get(cli_tree.LOOKUP_IS_HIDDEN, d.get('hidden', False))
    flag.hidden = flag.is_hidden
    flag.is_positional = False
    flag.is_required = d.get(cli_tree.LOOKUP_IS_REQUIRED, d.get(cli_tree.LOOKUP_REQUIRED, False))
    flag.required = flag.is_required
    flag.help = flag.description
    flag.dest = flag.name.lower().replace('-', '_')
    flag.metavar = flag.value
    flag.option_strings = [flag.name]
    if not hasattr(flag, 'default'):
        flag.default = None
    if flag.type == 'bool':
        flag.nargs = 0
    elif flag.nargs not in ('?', '*', '+'):
        flag.nargs = 1
    if flag.type == 'dict':
        flag.type = arg_parsers.ArgDict()
    elif flag.type == 'list':
        flag.type = arg_parsers.ArgList()
    elif flag.type == 'string':
        flag.type = None
    if flag.attr.get(cli_tree.LOOKUP_INVERTED_SYNOPSIS):
        flag.inverted_synopsis = True
    prop = flag.attr.get('property')
    if prop:
        if cli_tree.LOOKUP_VALUE in prop:
            kind = 'value'
            value = prop[cli_tree.LOOKUP_VALUE]
        else:
            value = None
            kind = 'bool' if flag.type == 'bool' else None
        flag.store_property = (properties.FromString(prop[cli_tree.LOOKUP_NAME]), kind, value)
    return flag