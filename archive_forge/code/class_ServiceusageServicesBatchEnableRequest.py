from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesBatchEnableRequest(_messages.Message):
    """A ServiceusageServicesBatchEnableRequest object.

  Fields:
    batchEnableServicesRequest: A BatchEnableServicesRequest resource to be
      passed as the request body.
    parent: Parent to enable services on. An example name would be:
      `projects/123` where `123` is the project number. The
      `BatchEnableServices` method currently only supports projects.
  """
    batchEnableServicesRequest = _messages.MessageField('BatchEnableServicesRequest', 1)
    parent = _messages.StringField(2, required=True)