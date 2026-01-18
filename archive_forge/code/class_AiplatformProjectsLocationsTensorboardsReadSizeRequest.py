from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsReadSizeRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsReadSizeRequest object.

  Fields:
    tensorboard: Required. The name of the Tensorboard resource. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`
  """
    tensorboard = _messages.StringField(1, required=True)