from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.api_lib.util import resource_search
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import completion_cache
from googlecloudsdk.core.cache import resource_cache
import six
class MultiResourceCompleter(Converter):
    """A completer that composes multiple resource completers.

  Attributes:
    completers: The list of completers.
  """

    def __init__(self, completers=None, qualified_parameter_names=None, **kwargs):
        """Constructor.

    Args:
      completers: The list of completers.
      qualified_parameter_names: The set of parameter names that must be
        qualified.
      **kwargs: Base class kwargs.
    """
        self.completers = [completer_class(**kwargs) for completer_class in completers]
        name_count = {}
        if qualified_parameter_names:
            for name in qualified_parameter_names:
                name_count[name] = 1
        for completer in self.completers:
            if completer.parameters:
                for parameter in completer.parameters:
                    if parameter.name in name_count:
                        name_count[parameter.name] += 1
                    else:
                        name_count[parameter.name] = 1
        qualified_parameter_names = {name for name, count in six.iteritems(name_count) if count != len(self.completers)}
        collections = []
        apis = set()
        for completer in self.completers:
            completer.AddQualifiedParameterNames(qualified_parameter_names)
            apis.add(completer.collection.split('.')[0])
            collections.append(completer.collection)
        collection = ','.join(collections)
        api = apis.pop() if len(apis) == 1 else None
        super(MultiResourceCompleter, self).__init__(collection=collection, api=api, **kwargs)

    def Complete(self, prefix, parameter_info):
        """Returns the union of completions from all completers."""
        return sorted({completions for completer in self.completers for completions in completer.Complete(prefix, parameter_info)})

    def Update(self, parameter_info, aggregations):
        """Update handled by self.completers."""
        del parameter_info
        del aggregations