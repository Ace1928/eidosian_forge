from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InventoryWindowsQuickFixEngineeringPackage(_messages.Message):
    """Information related to a Quick Fix Engineering package. Fields are taken
  from Windows QuickFixEngineering Interface and match the source names:
  https://docs.microsoft.com/en-
  us/windows/win32/cimwin32prov/win32-quickfixengineering

  Fields:
    caption: A short textual description of the QFE update.
    description: A textual description of the QFE update.
    hotFixId: Unique identifier associated with a particular QFE update.
    installTime: Date that the QFE update was installed. Mapped from
      installed_on field.
  """
    caption = _messages.StringField(1)
    description = _messages.StringField(2)
    hotFixId = _messages.StringField(3)
    installTime = _messages.StringField(4)