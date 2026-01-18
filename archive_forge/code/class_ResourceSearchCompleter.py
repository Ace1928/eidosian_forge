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
class ResourceSearchCompleter(ResourceCompleter):
    """A parameterized completer that uses Cloud Resource Search for updates."""

    def Update(self, parameter_info, aggregations):
        """Returns the current list of parsed resources."""
        query = '@type:{}'.format(self.collection)
        log.info('cloud resource search query: %s' % query)
        try:
            items = resource_search.List(query=query, uri=True)
        except Exception as e:
            if properties.VALUES.core.print_completion_tracebacks.GetBool():
                raise
            log.info(six.text_type(e).rstrip())
            raise type(e)('Update resource query [{}]: {}'.format(query, six.text_type(e).rstrip()))
        return [self.StringToRow(item, parameter_info) for item in items]