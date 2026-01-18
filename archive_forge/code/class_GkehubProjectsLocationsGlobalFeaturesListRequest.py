from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsGlobalFeaturesListRequest(_messages.Message):
    """A GkehubProjectsLocationsGlobalFeaturesListRequest object.

  Fields:
    filter: Lists Features that match the filter expression, following the
      syntax outlined in https://google.aip.dev/160. Examples: - Feature with
      the name "servicemesh" in project "foo-proj": name = "projects/foo-
      proj/locations/global/features/servicemesh" - Service Mesh Feature with
      `mtls` enabled: servicemesh_feature_spec.mtls = true - Features that
      have a label called `foo`: labels.foo:* - Features that have a label
      called `foo` whose value is `bar`: labels.foo = bar
    orderBy: One or more fields to compare and use to sort the output. See
      https://google.aip.dev/132#ordering.
    pageSize: When requesting a 'page' of resources, `page_size` specifies
      number of resources to return. If unspecified or set to 0, all resources
      will be returned.
    pageToken: Token returned by previous call to `ListFeatures` which
      specifies the position in the list from where to continue listing the
      resources.
    parent: Required. The parent (project and location) where the Features
      will be listed. Specified in the format `projects/*/locations/global`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)