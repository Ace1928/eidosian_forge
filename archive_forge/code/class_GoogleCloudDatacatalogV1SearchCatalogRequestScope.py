from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1SearchCatalogRequestScope(_messages.Message):
    """The criteria that select the subspace used for query matching.

  Fields:
    includeGcpPublicDatasets: If `true`, include Google Cloud public datasets
      in search results. By default, they are excluded. See [Google Cloud
      Public Datasets](/public-datasets) for more information.
    includeOrgIds: The list of organization IDs to search within. To find your
      organization ID, follow the steps from [Creating and managing
      organizations] (/resource-manager/docs/creating-managing-organization).
    includeProjectIds: The list of project IDs to search within. For more
      information on the distinction between project names, IDs, and numbers,
      see [Projects](/docs/overview/#projects).
    includePublicTagTemplates: Optional. This field is deprecated. The search
      mechanism for public and private tag templates is the same.
    restrictedLocations: Optional. The list of locations to search within. If
      empty, all locations are searched. Returns an error if any location in
      the list isn't one of the [Supported
      regions](https://cloud.google.com/data-
      catalog/docs/concepts/regions#supported_regions). If a location is
      unreachable, its name is returned in the
      `SearchCatalogResponse.unreachable` field. To get additional information
      on the error, repeat the search request and set the location name as the
      value of this parameter.
    starredOnly: Optional. If `true`, search only among starred entries. By
      default, all results are returned, starred or not.
  """
    includeGcpPublicDatasets = _messages.BooleanField(1)
    includeOrgIds = _messages.StringField(2, repeated=True)
    includeProjectIds = _messages.StringField(3, repeated=True)
    includePublicTagTemplates = _messages.BooleanField(4)
    restrictedLocations = _messages.StringField(5, repeated=True)
    starredOnly = _messages.BooleanField(6)