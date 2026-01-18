from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InventoryWindowsApplication(_messages.Message):
    """Contains information about a Windows application that is retrieved from
  the Windows Registry. For more information about these fields, see:
  https://docs.microsoft.com/en-us/windows/win32/msi/uninstall-registry-key

  Fields:
    displayName: The name of the application or product.
    displayVersion: The version of the product or application in string
      format.
    helpLink: The internet address for technical support.
    installDate: The last time this product received service. The value of
      this property is replaced each time a patch is applied or removed from
      the product or the command-line option is used to repair the product.
    publisher: The name of the manufacturer for the product or application.
  """
    displayName = _messages.StringField(1)
    displayVersion = _messages.StringField(2)
    helpLink = _messages.StringField(3)
    installDate = _messages.MessageField('Date', 4)
    publisher = _messages.StringField(5)