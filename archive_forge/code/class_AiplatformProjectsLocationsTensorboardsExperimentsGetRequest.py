from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsGetRequest object.

  Fields:
    name: Required. The name of the TensorboardExperiment resource. Format: `p
      rojects/{project}/locations/{location}/tensorboards/{tensorboard}/experi
      ments/{experiment}`
  """
    name = _messages.StringField(1, required=True)