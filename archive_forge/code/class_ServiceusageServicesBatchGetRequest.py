from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesBatchGetRequest(_messages.Message):
    """A ServiceusageServicesBatchGetRequest object.

  Fields:
    names: Names of the services to retrieve. An example name would be:
      `projects/123/services/serviceusage.googleapis.com` where `123` is the
      project number. A single request can get a maximum of 30 services at a
      time.
    parent: Parent to retrieve services from. If this is set, the parent of
      all of the services specified in `names` must match this field. An
      example name would be: `projects/123` where `123` is the project number.
      The `BatchGetServices` method currently only supports projects.
  """
    names = _messages.StringField(1, repeated=True)
    parent = _messages.StringField(2, required=True)