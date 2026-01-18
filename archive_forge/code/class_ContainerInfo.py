from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerInfo(_messages.Message):
    """Docker image that is used to create a container and start a VM instance
  for the version that you deploy. Only applicable for instances running in
  the App Engine flexible environment.

  Fields:
    image: URI to the hosted container image in Google Container Registry. The
      URI must be fully qualified and include a tag or digest. Examples:
      "gcr.io/my-project/image:tag" or "gcr.io/my-project/image@digest"
  """
    image = _messages.StringField(1)