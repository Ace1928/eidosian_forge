from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsAdaptiveMtDatasetsCreateRequest(_messages.Message):
    """A TranslateProjectsLocationsAdaptiveMtDatasetsCreateRequest object.

  Fields:
    adaptiveMtDataset: A AdaptiveMtDataset resource to be passed as the
      request body.
    parent: Required. Name of the parent project. In form of
      `projects/{project-number-or-id}/locations/{location-id}`
  """
    adaptiveMtDataset = _messages.MessageField('AdaptiveMtDataset', 1)
    parent = _messages.StringField(2, required=True)