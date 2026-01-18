from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirResourceIncomingReferencesRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirResourceIncomingRefer
  encesRequest object.

  Fields:
    _count: Maximum number of resources in a page. If not specified, 100 is
      used. May not be larger than 1000.
    _page_token: Used to retrieve the next page of results when using
      pagination. Set `_page_token` to the value of _page_token set in next
      page links' url. Next page are returned in the response bundle's links
      field, where `link.relation` is "next". Omit `_page_token` if no
      previous request has been made.
    _summary: Used to simplify the representation of the returned resources.
      `_summary=text` returns only the `text`, `id`, and `meta` top-level
      fields. `_summary=data` removes the `text` field and returns all other
      fields. `_summary=false` returns all parts of the resource(s). Either
      not providing this parameter or providing an empty value to this
      parameter also returns all parts of the resource(s).
    _type: String of comma-delimited FHIR resource types. If provided, only
      resources of the specified resource type(s) are returned. If not
      provided or an empty value is provided, no filter on the returned
      resource type(s) is applied.
    parent: Required. The name of the FHIR store that holds the target
      resource.
    target: Required. The target whose incoming references are requested. This
      param is required and must not be empty. It uses the format
      "ResourceType/ResourceID", for example, target=ResourceType/ResourceID.
  """
    _count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    _page_token = _messages.StringField(2)
    _summary = _messages.StringField(3)
    _type = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    target = _messages.StringField(6)