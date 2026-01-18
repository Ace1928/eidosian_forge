from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CrashDialogError(_messages.Message):
    """Crash dialog was detected during the test execution

  Fields:
    crashPackage: The name of the package that caused the dialog.
  """
    crashPackage = _messages.StringField(1)