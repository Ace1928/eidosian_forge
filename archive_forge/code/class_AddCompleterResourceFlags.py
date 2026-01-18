from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.meta import cache_util
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core.console import console_io
import six
class AddCompleterResourceFlags(parser_extensions.DynamicPositionalAction):
    """Adds resource argument flags based on the completer."""

    def __init__(self, *args, **kwargs):
        super(AddCompleterResourceFlags, self).__init__(*args, **kwargs)
        self.__argument = None
        self.__completer = None

    def GenerateArgs(self, namespace, module_path):
        args = []
        presentation_kwargs = namespace.resource_presentation_kwargs or {}
        if namespace.resource_spec_path:
            spec = _GetPresentationSpec(namespace.resource_spec_path, **presentation_kwargs)
            info = concept_parsers.ConceptParser([spec]).GetInfo(spec.name)
            for arg in info.GetAttributeArgs():
                if arg.name.startswith('--'):
                    arg.kwargs['required'] = False
                else:
                    arg.kwargs['nargs'] = '?' if not spec.plural else '*'
                args.append(arg)
        kwargs = namespace.kwargs or {}
        self.__completer = _GetCompleter(module_path, qualify=namespace.qualify, resource_spec=namespace.resource_spec_path, presentation_kwargs=presentation_kwargs, attribute=namespace.attribute, **kwargs)
        if self.__completer.parameters:
            for parameter in self.__completer.parameters:
                dest = parameter_info_lib.GetDestFromParam(parameter.name)
                if hasattr(namespace, dest):
                    continue
                flag = parameter_info_lib.GetFlagFromDest(dest)
                arg = base.Argument(flag, dest=dest, category='RESOURCE COMPLETER', help='{} `{}` parameter value.'.format(self.__completer.__class__.__name__, parameter.name))
                args.append(arg)
        self.__argument = base.Argument('resource_to_complete', nargs='?', help='The partial resource name to complete. Omit to enter an interactive loop that reads a partial resource name from the input and lists the possible prefix matches on the output or displays an ERROR message.')
        args.append(self.__argument)
        return args

    def Completions(self, prefix, parsed_args, **kwargs):
        parameter_info = self.__completer.ParameterInfo(parsed_args, self.__argument)
        return self.__completer.Complete(prefix, parameter_info)