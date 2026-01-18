from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityProfileScoringConfig(_messages.Message):
    """Security configurations to manage scoring.

  Fields:
    description: Description of the config.
    scorePath: Path of the component config used for scoring.
    title: Title of the config.
  """
    description = _messages.StringField(1)
    scorePath = _messages.StringField(2)
    title = _messages.StringField(3)