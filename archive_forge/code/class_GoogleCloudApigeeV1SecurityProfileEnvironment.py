from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityProfileEnvironment(_messages.Message):
    """Environment information of attached environments. Scoring an environment
  is enabled only if it is attached to a security profile.

  Fields:
    attachTime: Output only. Time at which environment was attached to the
      security profile.
    environment: Output only. Name of the environment.
  """
    attachTime = _messages.StringField(1)
    environment = _messages.StringField(2)