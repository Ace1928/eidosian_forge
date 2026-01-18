from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyConfig(_messages.Message):
    """Message describing PolicyConfig object consisting of policy name and its
  configurations.

  Fields:
    policy: Required. Full policy name. Example:
      projects/{project}/locations/{location}/policies/{policy}
  """
    policy = _messages.StringField(1)