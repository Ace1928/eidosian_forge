from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsTransformersGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsTransformersGetRequest object.

  Fields:
    name: Required. The name of the transformer to retrieve, in the following
      form:
      `projects/{project}/locations/{location}/transformers/{transformer}`.
  """
    name = _messages.StringField(1, required=True)