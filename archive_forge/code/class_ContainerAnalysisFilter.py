from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class ContainerAnalysisFilter:
    """Utility class for creating filters to send to containeranalysis API.

  If passed to a request, only occurrences that have the resource prefix, is of
  one of the kinds in self._kinds, is for one of the resources in self._resource
  and satisfies self._custom_filter will be retrieved.

  Properties:
    resource_prefixes: list, the resource prefixes filter added to this filter.
    custom_filter: str, the user provided filter added to this filter.
    kinds: list, metadata kinds added to this filter.
    resources: list, resource URLs added to this filter.
  """

    def __init__(self, max_resource_chunk_size=_DEFAULT_RESOURCE_URI_CHUNK_SIZE):
        self._resource_prefixes = []
        self._custom_filter = ''
        self._kinds = []
        self._resources = []
        self._max_resource_chunk_size = max_resource_chunk_size

    @property
    def resource_prefixes(self):
        return self._resource_prefixes

    @property
    def custom_filter(self):
        return self._custom_filter

    @property
    def kinds(self):
        return self._kinds

    @property
    def resources(self):
        return self._resources

    def WithKinds(self, kinds):
        """Add metadata kinds to this filter."""
        self._kinds = list(kinds)
        return self

    def WithResources(self, resources):
        """Add resources to this filter."""
        self._resources = list(resources)
        return self

    def WithCustomFilter(self, custom_filter):
        """Add a custom filter to this filter."""
        self._custom_filter = custom_filter
        return self

    def WithResourcePrefixes(self, resource_prefixes):
        """Add resource prefixes to this filter."""
        self._resource_prefixes = list(resource_prefixes)
        return self

    def GetFilter(self):
        """Returns a filter string with filtering attributes set."""
        kinds = _OrJoinFilters(*[_HasField('kind', k) for k in self._kinds])
        resources = _OrJoinFilters(*[_HasField('resourceUrl', r) for r in self._resources])
        return _AndJoinFilters(_HasPrefixes('resourceUrl', self.resource_prefixes), self.custom_filter, kinds, resources)

    def GetChunkifiedFilters(self):
        """Returns a list of filter strings where each filter has an upper limit of resource filters.

    The upper limit of resource filters in a contructed filter string is set
    by self._max_resource_chunk_size. This is to avoid having too many
    filters in one API request and getting the request rejected.


    For example, consider this ContainerAnalysisFilter object:
      ContainerAnalysisFilter() \\
        .WithKinds('VULNERABILITY') \\
        .WithResources([
          'url/to/resources/1', 'url/to/resources/2', 'url/to/resources/3',
          'url/to/resources/4', 'url/to/resources/5', 'url/to/resources/6'])

    Calling GetChunkifiedFilters will return the following result:
    [
      '''(kind="VULNERABILITY") AND (resource_url="'url/to/resources/1)"
       OR ("resource_url="'url/to/resources/2")
       OR ("resource_url="'url/to/resources/3")
       OR ("resource_url="'url/to/resources/4")
       OR ("resource_url="'url/to/resources/5")''',
      '(kind="VULNERABILITY") AND (resource_url="url/to/resources/6")'
    ]
    """
        kinds = _OrJoinFilters(*[_HasField('kind', k) for k in self._kinds])
        resources = [_HasField('resourceUrl', r) for r in self._resources]
        base_filter = _AndJoinFilters(_HasPrefixes('resourceUrl', self.resource_prefixes), self.custom_filter, kinds)
        if not resources:
            return [base_filter]
        chunks = [resources[i:i + self._max_resource_chunk_size] for i in range(0, len(resources), self._max_resource_chunk_size)]
        return [_AndJoinFilters(base_filter, _OrJoinFilters(*chunk)) for chunk in chunks]