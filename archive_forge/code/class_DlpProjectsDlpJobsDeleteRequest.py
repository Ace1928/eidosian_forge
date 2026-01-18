from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsDlpJobsDeleteRequest(_messages.Message):
    """A DlpProjectsDlpJobsDeleteRequest object.

  Fields:
    name: Required. The name of the DlpJob resource to be deleted.
  """
    name = _messages.StringField(1, required=True)