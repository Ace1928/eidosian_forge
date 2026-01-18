from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FetchInventoryResponse(_messages.Message):
    """Response message for fetchInventory.

  Fields:
    awsVms: The description of the VMs in a Source of type AWS.
    azureVms: The description of the VMs in a Source of type Azure.
    nextPageToken: Output only. A token, which can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    updateTime: Output only. The timestamp when the source was last queried
      (if the result is from the cache).
    vmwareVms: The description of the VMs in a Source of type Vmware.
  """
    awsVms = _messages.MessageField('AwsVmsDetails', 1)
    azureVms = _messages.MessageField('AzureVmsDetails', 2)
    nextPageToken = _messages.StringField(3)
    updateTime = _messages.StringField(4)
    vmwareVms = _messages.MessageField('VmwareVmsDetails', 5)