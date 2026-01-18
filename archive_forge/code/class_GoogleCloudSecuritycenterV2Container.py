from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Container(_messages.Message):
    """Container associated with the finding.

  Fields:
    createTime: The time that the container was created.
    imageId: Optional container image ID, if provided by the container
      runtime. Uniquely identifies the container image launched using a
      container image digest.
    labels: Container labels, as provided by the container runtime.
    name: Name of the container.
    uri: Container image URI provided when configuring a pod or container.
      This string can identify a container image version using mutable tags.
  """
    createTime = _messages.StringField(1)
    imageId = _messages.StringField(2)
    labels = _messages.MessageField('GoogleCloudSecuritycenterV2Label', 3, repeated=True)
    name = _messages.StringField(4)
    uri = _messages.StringField(5)