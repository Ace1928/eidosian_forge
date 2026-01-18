from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogErrorGroup(_messages.Message):
    """Contains metadata that associates the LogEntry to Error Reporting error
  groups.

  Fields:
    id: The id is a unique identifier for a particular error group; it is the
      last part of the error group resource name:
      /project/[PROJECT_ID]/errors/[ERROR_GROUP_ID]. Example: COShysOX0r_51QE.
      The id is derived from key parts of the error-log content and is treated
      as Service Data. For information about how Service Data is handled, see
      Google Cloud Privacy Notice (https://cloud.google.com/terms/cloud-
      privacy-notice).
  """
    id = _messages.StringField(1)