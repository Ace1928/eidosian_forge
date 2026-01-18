from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmImage(_messages.Message):
    """Definition of a custom Compute Engine virtual machine image for starting
  a notebook instance with the environment installed directly on the VM.

  Fields:
    family: Optional. Use this VM image family to find the image; the newest
      image in this family will be used.
    name: Optional. Use VM image name to find the image.
    project: Required. The name of the Google Cloud project that this VM image
      belongs to. Format: `{project_id}`
  """
    family = _messages.StringField(1)
    name = _messages.StringField(2)
    project = _messages.StringField(3)