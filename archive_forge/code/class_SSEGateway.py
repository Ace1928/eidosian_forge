from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SSEGateway(_messages.Message):
    """Message describing SSEGateway object

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    appFacingNetwork: Output only. [Output only] SSE-owned network which the
      app network should peer with.
    appFacingSubnetRange: Optional. Subnet range of the SSE-owned subnet where
      app traffic is routed. Defaults to "100.64.1.0/24" if unspecified. The
      CIDR suffix should be less than or equal to 24.
    appFacingTargetIp: Optional. Target IP where app traffic is routed.
      Defaults to "100.64.1.253" if unspecified.
    appNetworks: Optional. List of app networks which are attached to this
      gateway.
    country: Optional. ISO-3166 alpha 2 country code used for localization.
      Only used for Symantec's API today, and is optional even for gateways
      connected to Symantec, since Symantec applies a default if we don't
      specify it. Not case-sensitive, since it will be upper-cased when
      sending to Symantec API.
    createTime: Output only. [Output only] Create time stamp
    labels: Optional. Labels as key value pairs
    maxBandwidthMbps: Optional. Not an enforced cap. Used only for Symantec's
      API today.
    name: Immutable. name of resource
    sseProject: Output only. [Output Only] The project owning
      app_facing_network and untrusted_facing_network
    sseRealm: Required. ID of SSERealm owning the SSEGateway
    symantecOptions: Optional. Required iff the associated realm is of type
      SYMANTEC_CLOUD_SWG.
    timezone: Optional. tzinfo identifier used for localization. Only used for
      Symantec's API today, and is optional even for gateways connected to
      Symantec, since Symantec applies a default if we don't specify it. Case
      sensitive.
    untrustedFacingNetwork: Output only. [Output only] SSE-owned network which
      the untrusted network should peer with.
    untrustedFacingSubnetRange: Optional. Subnet range of the SSE-owned subnet
      where untrusted traffic is routed. Defaults to "100.64.2.0/24" if
      unspecified. The CIDR suffix should be less than or equal to 24.
    untrustedFacingTargetIp: Optional. Target IP where untrusted traffic is
      routed. Default value is set to "100.64.2.253".
    untrustedNetwork: Optional. Customer-owned network where untrusted users
      land.
    updateTime: Output only. [Output only] Update time stamp
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    appFacingNetwork = _messages.StringField(1)
    appFacingSubnetRange = _messages.StringField(2)
    appFacingTargetIp = _messages.StringField(3)
    appNetworks = _messages.StringField(4, repeated=True)
    country = _messages.StringField(5)
    createTime = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    maxBandwidthMbps = _messages.IntegerField(8)
    name = _messages.StringField(9)
    sseProject = _messages.StringField(10)
    sseRealm = _messages.StringField(11)
    symantecOptions = _messages.MessageField('SSEGatewaySSEGatewaySymantecOptions', 12)
    timezone = _messages.StringField(13)
    untrustedFacingNetwork = _messages.StringField(14)
    untrustedFacingSubnetRange = _messages.StringField(15)
    untrustedFacingTargetIp = _messages.StringField(16)
    untrustedNetwork = _messages.StringField(17)
    updateTime = _messages.StringField(18)