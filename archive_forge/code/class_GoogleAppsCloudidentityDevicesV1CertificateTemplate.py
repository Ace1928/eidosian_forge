from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1CertificateTemplate(_messages.Message):
    """CertificateTemplate (v3 Extension in X.509).

  Fields:
    id: The template id of the template. Example: "1.3.6.1.4.1.311.21.8.156086
      21.11768144.5720724.16068415.6889630.81.2472537.7784047".
    majorVersion: The Major version of the template. Example: 100.
    minorVersion: The minor version of the template. Example: 12.
  """
    id = _messages.StringField(1)
    majorVersion = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    minorVersion = _messages.IntegerField(3, variant=_messages.Variant.INT32)