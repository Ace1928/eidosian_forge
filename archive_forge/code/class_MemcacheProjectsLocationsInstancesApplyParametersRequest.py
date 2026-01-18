from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemcacheProjectsLocationsInstancesApplyParametersRequest(_messages.Message):
    """A MemcacheProjectsLocationsInstancesApplyParametersRequest object.

  Fields:
    applyParametersRequest: A ApplyParametersRequest resource to be passed as
      the request body.
    name: Required. Resource name of the Memcached instance for which
      parameter group updates should be applied.
  """
    applyParametersRequest = _messages.MessageField('ApplyParametersRequest', 1)
    name = _messages.StringField(2, required=True)