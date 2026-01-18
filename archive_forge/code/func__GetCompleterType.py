from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.calliope import walker
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import exceptions as cache_exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.cache import resource_cache
import six
def _GetCompleterType(completer_class):
    """Returns the completer type name given its class."""
    completer_type = None
    try:
        for t in completer_class.mro():
            if t == completers.ResourceCompleter:
                break
            if t.__name__.endswith('Completer'):
                completer_type = t.__name__
    except AttributeError:
        pass
    if not completer_type and callable(completer_class):
        completer_type = 'function'
    return completer_type