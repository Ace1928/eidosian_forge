from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateProviderInstance(_messages.Message):
    """Specification of a TLS certificate provider instance. Workloads may have
  one or more CertificateProvider instances (plugins) and one of them is
  enabled and configured by specifying this message. Workloads use the values
  from this message to locate and load the CertificateProvider instance
  configuration.

  Fields:
    pluginInstance: Required. Plugin instance name, used to locate and load
      CertificateProvider instance configuration. Set to
      "google_cloud_private_spiffe" to use Certificate Authority Service
      certificate provider instance.
  """
    pluginInstance = _messages.StringField(1)