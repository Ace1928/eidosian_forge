from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsWorkloadsListRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsWorkloadsListRequest object.

  Fields:
    filter: Optional. The list filter. Currently only supports equality on the
      type field. The value of a field specified in the filter expression must
      be one ComposerWorkloadType enum option. It's possible to get multiple
      types using "OR" operator, e.g.: "type=SCHEDULER OR type=CELERY_WORKER".
      If not specified, all items are returned.
    pageSize: Optional. The maximum number of environments to return.
    pageToken: Optional. The next_page_token value returned from a previous
      List request, if any.
    parent: Required. The environment name to get workloads for, in the form:
      "projects/{projectId}/locations/{locationId}/environments/{environmentId
      }"
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)