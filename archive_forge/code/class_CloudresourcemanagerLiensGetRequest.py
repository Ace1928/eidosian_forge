from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerLiensGetRequest(_messages.Message):
    """A CloudresourcemanagerLiensGetRequest object.

  Fields:
    liensId: Part of `name`. Required. The name/identifier of the Lien.
  """
    liensId = _messages.StringField(1, required=True)