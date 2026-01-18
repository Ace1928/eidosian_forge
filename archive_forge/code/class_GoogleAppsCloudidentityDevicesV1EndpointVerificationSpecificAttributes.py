from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1EndpointVerificationSpecificAttributes(_messages.Message):
    """Resource representing the [Endpoint Verification-specific
  attributes](https://cloud.google.com/endpoint-verification/docs/device-
  information) of a device.

  Messages:
    AdditionalSignalsValue: Additional signals reported by Endpoint
      Verification. It includes the following attributes: 1. Non-configurable
      attributes: hotfixes, av_installed, av_enabled, windows_domain_name,
      is_os_native_firewall_enabled, and is_secure_boot_enabled. 2.
      [Configurable attributes](https://cloud.google.com/endpoint-
      verification/docs/collect-config-attributes): file, folder, and binary
      attributes; registry entries; and properties in a plist.

  Fields:
    additionalSignals: Additional signals reported by Endpoint Verification.
      It includes the following attributes: 1. Non-configurable attributes:
      hotfixes, av_installed, av_enabled, windows_domain_name,
      is_os_native_firewall_enabled, and is_secure_boot_enabled. 2.
      [Configurable attributes](https://cloud.google.com/endpoint-
      verification/docs/collect-config-attributes): file, folder, and binary
      attributes; registry entries; and properties in a plist.
    browserAttributes: Details of browser profiles reported by Endpoint
      Verification.
    certificateAttributes: Details of certificates.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AdditionalSignalsValue(_messages.Message):
        """Additional signals reported by Endpoint Verification. It includes the
    following attributes: 1. Non-configurable attributes: hotfixes,
    av_installed, av_enabled, windows_domain_name,
    is_os_native_firewall_enabled, and is_secure_boot_enabled. 2.
    [Configurable attributes](https://cloud.google.com/endpoint-
    verification/docs/collect-config-attributes): file, folder, and binary
    attributes; registry entries; and properties in a plist.

    Messages:
      AdditionalProperty: An additional property for a AdditionalSignalsValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AdditionalSignalsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    additionalSignals = _messages.MessageField('AdditionalSignalsValue', 1)
    browserAttributes = _messages.MessageField('GoogleAppsCloudidentityDevicesV1BrowserAttributes', 2, repeated=True)
    certificateAttributes = _messages.MessageField('GoogleAppsCloudidentityDevicesV1CertificateAttributes', 3, repeated=True)