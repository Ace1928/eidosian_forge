from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryGroupsEntriesCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryGroupsEntriesCreateRequest object.

  Fields:
    entryId: Required. Entry identifier. It has to be unique within an Entry
      Group.Entries corresponding to Google Cloud resources use Entry ID
      format based on Full Resource Names (https://cloud.google.com/apis/desig
      n/resource_names#full_resource_name). The format is a Full Resource Name
      of the resource without the prefix double slashes in the API Service
      Name part of Full Resource Name. This allows retrieval of entries using
      their associated resource name.For example if the Full Resource Name of
      a resource is //library.googleapis.com/shelves/shelf1/books/book2, then
      the suggested entry_id is
      library.googleapis.com/shelves/shelf1/books/book2.It is also suggested
      to follow the same convention for entries corresponding to resources
      from other providers or systems than Google Cloud.The maximum size of
      the field is 4000 characters.
    googleCloudDataplexV1Entry: A GoogleCloudDataplexV1Entry resource to be
      passed as the request body.
    parent: Required. The resource name of the parent Entry Group:
      projects/{project}/locations/{location}/entryGroups/{entry_group}.
  """
    entryId = _messages.StringField(1)
    googleCloudDataplexV1Entry = _messages.MessageField('GoogleCloudDataplexV1Entry', 2)
    parent = _messages.StringField(3, required=True)