from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetSslProxiesSetCertificateMapRequest(_messages.Message):
    """A TargetSslProxiesSetCertificateMapRequest object.

  Fields:
    certificateMap: URL of the Certificate Map to associate with this
      TargetSslProxy. Accepted format is
      //certificatemanager.googleapis.com/projects/{project
      }/locations/{location}/certificateMaps/{resourceName}.
  """
    certificateMap = _messages.StringField(1)