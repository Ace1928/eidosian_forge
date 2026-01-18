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
def _GetCompleter(module_path, cache=None, qualify=None, resource_spec=None, presentation_kwargs=None, attribute=None, **kwargs):
    """Returns an instantiated completer for module_path."""
    presentation_kwargs = presentation_kwargs or {}
    if resource_spec:
        presentation_spec = _GetPresentationSpec(resource_spec, **presentation_kwargs)
        completer = module_util.ImportModule(module_path)(presentation_spec.concept_spec, attribute)
    else:
        completer = module_util.ImportModule(module_path)
        if not isinstance(completer, type):
            return _FunctionCompleter(completer)
    try:
        return completer(cache=cache, qualified_parameter_names=qualify, **kwargs)
    except TypeError:
        return _FunctionCompleter(completer())