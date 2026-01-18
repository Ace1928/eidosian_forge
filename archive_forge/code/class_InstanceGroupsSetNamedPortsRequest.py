from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupsSetNamedPortsRequest(_messages.Message):
    """A InstanceGroupsSetNamedPortsRequest object.

  Fields:
    fingerprint: The fingerprint of the named ports information for this
      instance group. Use this optional property to prevent conflicts when
      multiple users change the named ports settings concurrently. Obtain the
      fingerprint with the instanceGroups.get method. Then, include the
      fingerprint in your request to ensure that you do not overwrite changes
      that were applied from another concurrent request. A request with an
      incorrect fingerprint will fail with error 412 conditionNotMet.
    namedPorts: The list of named ports to set for this instance group.
  """
    fingerprint = _messages.BytesField(1)
    namedPorts = _messages.MessageField('NamedPort', 2, repeated=True)