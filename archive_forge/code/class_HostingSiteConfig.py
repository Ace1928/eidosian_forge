from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HostingSiteConfig(_messages.Message):
    """Message for defining the firebase hosting site configuration.

  Fields:
    site_id: Firebase hosting site-ID.
  """
    site_id = _messages.StringField(1)