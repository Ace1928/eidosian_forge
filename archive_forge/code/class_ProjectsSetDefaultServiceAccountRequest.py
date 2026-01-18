from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectsSetDefaultServiceAccountRequest(_messages.Message):
    """A ProjectsSetDefaultServiceAccountRequest object.

  Fields:
    email: Email address of the service account.
  """
    email = _messages.StringField(1)