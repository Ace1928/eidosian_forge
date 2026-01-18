from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import completers
from googlecloudsdk.core.util import text
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def _KwargsForAttribute(self, name, attribute):
    """Constructs the kwargs for adding an attribute to argparse."""
    required = self._IsRequiredArg(attribute)
    final_help_text = self._GetHelpTextForAttribute(attribute)
    plural = self._IsPluralArg(attribute)
    if attribute.completer:
        completer = attribute.completer
    elif not self.resource_spec.disable_auto_completers:
        completer = completers.CompleterForAttribute(self.resource_spec, attribute.name)
    else:
        completer = None
    kwargs_dict = {'help': final_help_text, 'type': attribute.value_type, 'completer': completer, 'hidden': self.hidden}
    if util.IsPositional(name):
        if plural and required:
            kwargs_dict.update({'nargs': '+'})
        elif plural and (not required):
            kwargs_dict.update({'nargs': '*'})
        elif not required:
            kwargs_dict.update({'nargs': '?'})
    else:
        kwargs_dict.update({'metavar': util.MetavarFormat(name)})
        if required:
            kwargs_dict.update({'required': True})
        if plural:
            kwargs_dict.update({'type': arg_parsers.ArgList()})
    return kwargs_dict