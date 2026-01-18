from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsRolloutsListRequest(_messages.Message):
    """A GkehubProjectsLocationsRolloutsListRequest object.

  Fields:
    filter: Optional. Lists Rollouts that match the filter expression,
      following the syntax outlined in https://google.aip.dev/160.
    pageSize: The maximum number of rollout to return. The service may return
      fewer than this value. If unspecified, at most 50 rollouts will be
      returned. The maximum value is 1000; values above 1000 will be coerced
      to 1000.
    pageToken: A page token, received from a previous `ListRollouts` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListRollouts` must match the call that provided
      the page token.
    parent: Required. The parent, which owns this collection of rollout.
      Format: projects/{project}/locations/{location}
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)