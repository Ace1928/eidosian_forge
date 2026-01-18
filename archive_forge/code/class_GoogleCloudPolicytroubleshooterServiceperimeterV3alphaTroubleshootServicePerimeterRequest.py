from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterServiceperimeterV3alphaTroubleshootServicePerimeterRequest(_messages.Message):
    """LINT.IfChange Request to troubleshoot service perimeters

  Fields:
    troubleshootingToken: The troubleshooting token can be generated when
      customers get access denied by the service perimeter
  """
    troubleshootingToken = _messages.StringField(1)