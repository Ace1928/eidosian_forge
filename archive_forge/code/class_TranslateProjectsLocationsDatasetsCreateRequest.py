from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsDatasetsCreateRequest(_messages.Message):
    """A TranslateProjectsLocationsDatasetsCreateRequest object.

  Fields:
    dataset: A Dataset resource to be passed as the request body.
    parent: Required. The project name.
  """
    dataset = _messages.MessageField('Dataset', 1)
    parent = _messages.StringField(2, required=True)