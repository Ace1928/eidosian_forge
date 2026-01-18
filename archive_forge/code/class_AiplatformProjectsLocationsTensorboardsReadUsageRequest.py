from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsReadUsageRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsReadUsageRequest object.

  Fields:
    tensorboard: Required. The name of the Tensorboard resource. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`
  """
    tensorboard = _messages.StringField(1, required=True)