from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2alpha1IosKeyRestrictions(_messages.Message):
    """Key restrictions that are specific to iOS keys.

  Fields:
    allowedBundleIds: A list of bundle IDs that are allowed when making API
      calls with this key.
  """
    allowedBundleIds = _messages.StringField(1, repeated=True)