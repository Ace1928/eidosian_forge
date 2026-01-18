from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Interconnect(_messages.Message):
    """Represents an Interconnect resource. An Interconnect resource is a
  dedicated connection between the Google Cloud network and your on-premises
  network. For more information, read the Dedicated Interconnect Overview.

  Enums:
    AvailableFeaturesValueListEntryValuesEnum:
    InterconnectTypeValueValuesEnum: Type of interconnect, which can take one
      of the following values: - PARTNER: A partner-managed interconnection
      shared between customers though a partner. - DEDICATED: A dedicated
      physical interconnection with the customer. Note that a value IT_PRIVATE
      has been deprecated in favor of DEDICATED.
    LinkTypeValueValuesEnum: Type of link requested, which can take one of the
      following values: - LINK_TYPE_ETHERNET_10G_LR: A 10G Ethernet with LR
      optics - LINK_TYPE_ETHERNET_100G_LR: A 100G Ethernet with LR optics.
      Note that this field indicates the speed of each of the links in the
      bundle, not the speed of the entire bundle.
    OperationalStatusValueValuesEnum: [Output Only] The current status of this
      Interconnect's functionality, which can take one of the following
      values: - OS_ACTIVE: A valid Interconnect, which is turned up and is
      ready to use. Attachments may be provisioned on this Interconnect. -
      OS_UNPROVISIONED: An Interconnect that has not completed turnup. No
      attachments may be provisioned on this Interconnect. -
      OS_UNDER_MAINTENANCE: An Interconnect that is undergoing internal
      maintenance. No attachments may be provisioned or updated on this
      Interconnect.
    RequestedFeaturesValueListEntryValuesEnum:
    StateValueValuesEnum: [Output Only] The current state of Interconnect
      functionality, which can take one of the following values: - ACTIVE: The
      Interconnect is valid, turned up and ready to use. Attachments may be
      provisioned on this Interconnect. - UNPROVISIONED: The Interconnect has
      not completed turnup. No attachments may be provisioned on this
      Interconnect. - UNDER_MAINTENANCE: The Interconnect is undergoing
      internal maintenance. No attachments may be provisioned or updated on
      this Interconnect.

  Messages:
    LabelsValue: Labels for this resource. These can only be added or modified
      by the setLabels method. Each label key/value pair must comply with
      RFC1035. Label values may be empty.

  Fields:
    adminEnabled: Administrative status of the interconnect. When this is set
      to true, the Interconnect is functional and can carry traffic. When set
      to false, no packets can be carried over the interconnect and no BGP
      routes are exchanged over it. By default, the status is set to true.
    availableFeatures: [Output only] List of features available for this
      Interconnect connection, which can take one of the following values: -
      MACSEC If present then the Interconnect connection is provisioned on
      MACsec capable hardware ports. If not present then the Interconnect
      connection is provisioned on non-MACsec capable ports and MACsec isn't
      supported and enabling MACsec fails.
    circuitInfos: [Output Only] A list of CircuitInfo objects, that describe
      the individual circuits in this LAG.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    customerName: Customer name, to put in the Letter of Authorization as the
      party authorized to request a crossconnect.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    expectedOutages: [Output Only] A list of outages expected for this
      Interconnect.
    googleIpAddress: [Output Only] IP address configured on the Google side of
      the Interconnect link. This can be used only for ping tests.
    googleReferenceId: [Output Only] Google reference ID to be used when
      raising support tickets with Google or otherwise to debug backend
      connectivity issues.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    interconnectAttachments: [Output Only] A list of the URLs of all
      InterconnectAttachments configured to use this Interconnect.
    interconnectType: Type of interconnect, which can take one of the
      following values: - PARTNER: A partner-managed interconnection shared
      between customers though a partner. - DEDICATED: A dedicated physical
      interconnection with the customer. Note that a value IT_PRIVATE has been
      deprecated in favor of DEDICATED.
    kind: [Output Only] Type of the resource. Always compute#interconnect for
      interconnects.
    labelFingerprint: A fingerprint for the labels being applied to this
      Interconnect, which is essentially a hash of the labels set used for
      optimistic locking. The fingerprint is initially generated by Compute
      Engine and changes after every request to modify or update labels. You
      must always provide an up-to-date fingerprint hash in order to update or
      change labels, otherwise the request will fail with error 412
      conditionNotMet. To see the latest fingerprint, make a get() request to
      retrieve an Interconnect.
    labels: Labels for this resource. These can only be added or modified by
      the setLabels method. Each label key/value pair must comply with
      RFC1035. Label values may be empty.
    linkType: Type of link requested, which can take one of the following
      values: - LINK_TYPE_ETHERNET_10G_LR: A 10G Ethernet with LR optics -
      LINK_TYPE_ETHERNET_100G_LR: A 100G Ethernet with LR optics. Note that
      this field indicates the speed of each of the links in the bundle, not
      the speed of the entire bundle.
    location: URL of the InterconnectLocation object that represents where
      this connection is to be provisioned.
    macsec: Configuration that enables Media Access Control security (MACsec)
      on the Cloud Interconnect connection between Google and your on-premises
      router.
    macsecEnabled: Enable or disable MACsec on this Interconnect connection.
      MACsec enablement fails if the MACsec object is not specified.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    nocContactEmail: Email address to contact the customer NOC for operations
      and maintenance notifications regarding this Interconnect. If specified,
      this will be used for notifications in addition to all other forms
      described, such as Cloud Monitoring logs alerting and Cloud
      Notifications. This field is required for users who sign up for Cloud
      Interconnect using workforce identity federation.
    operationalStatus: [Output Only] The current status of this Interconnect's
      functionality, which can take one of the following values: - OS_ACTIVE:
      A valid Interconnect, which is turned up and is ready to use.
      Attachments may be provisioned on this Interconnect. - OS_UNPROVISIONED:
      An Interconnect that has not completed turnup. No attachments may be
      provisioned on this Interconnect. - OS_UNDER_MAINTENANCE: An
      Interconnect that is undergoing internal maintenance. No attachments may
      be provisioned or updated on this Interconnect.
    peerIpAddress: [Output Only] IP address configured on the customer side of
      the Interconnect link. The customer should configure this IP address
      during turnup when prompted by Google NOC. This can be used only for
      ping tests.
    provisionedLinkCount: [Output Only] Number of links actually provisioned
      in this interconnect.
    remoteLocation: Indicates that this is a Cross-Cloud Interconnect. This
      field specifies the location outside of Google's network that the
      interconnect is connected to.
    requestedFeatures: Optional. List of features requested for this
      Interconnect connection, which can take one of the following values: -
      MACSEC If specified then the connection is created on MACsec capable
      hardware ports. If not specified, the default value is false, which
      allocates non-MACsec capable ports first if available. This parameter
      can be provided only with Interconnect INSERT. It isn't valid for
      Interconnect PATCH.
    requestedLinkCount: Target number of physical links in the link bundle, as
      requested by the customer.
    satisfiesPzs: [Output Only] Reserved for future use.
    selfLink: [Output Only] Server-defined URL for the resource.
    state: [Output Only] The current state of Interconnect functionality,
      which can take one of the following values: - ACTIVE: The Interconnect
      is valid, turned up and ready to use. Attachments may be provisioned on
      this Interconnect. - UNPROVISIONED: The Interconnect has not completed
      turnup. No attachments may be provisioned on this Interconnect. -
      UNDER_MAINTENANCE: The Interconnect is undergoing internal maintenance.
      No attachments may be provisioned or updated on this Interconnect.
  """

    class AvailableFeaturesValueListEntryValuesEnum(_messages.Enum):
        """AvailableFeaturesValueListEntryValuesEnum enum type.

    Values:
      IF_MACSEC: Media Access Control security (MACsec)
    """
        IF_MACSEC = 0

    class InterconnectTypeValueValuesEnum(_messages.Enum):
        """Type of interconnect, which can take one of the following values: -
    PARTNER: A partner-managed interconnection shared between customers though
    a partner. - DEDICATED: A dedicated physical interconnection with the
    customer. Note that a value IT_PRIVATE has been deprecated in favor of
    DEDICATED.

    Values:
      DEDICATED: A dedicated physical interconnection with the customer.
      IT_PRIVATE: [Deprecated] A private, physical interconnection with the
        customer.
      PARTNER: A partner-managed interconnection shared between customers via
        partner.
    """
        DEDICATED = 0
        IT_PRIVATE = 1
        PARTNER = 2

    class LinkTypeValueValuesEnum(_messages.Enum):
        """Type of link requested, which can take one of the following values: -
    LINK_TYPE_ETHERNET_10G_LR: A 10G Ethernet with LR optics -
    LINK_TYPE_ETHERNET_100G_LR: A 100G Ethernet with LR optics. Note that this
    field indicates the speed of each of the links in the bundle, not the
    speed of the entire bundle.

    Values:
      LINK_TYPE_ETHERNET_100G_LR: 100G Ethernet, LR Optics.
      LINK_TYPE_ETHERNET_10G_LR: 10G Ethernet, LR Optics. [(rate_bps) =
        10000000000];
    """
        LINK_TYPE_ETHERNET_100G_LR = 0
        LINK_TYPE_ETHERNET_10G_LR = 1

    class OperationalStatusValueValuesEnum(_messages.Enum):
        """[Output Only] The current status of this Interconnect's functionality,
    which can take one of the following values: - OS_ACTIVE: A valid
    Interconnect, which is turned up and is ready to use. Attachments may be
    provisioned on this Interconnect. - OS_UNPROVISIONED: An Interconnect that
    has not completed turnup. No attachments may be provisioned on this
    Interconnect. - OS_UNDER_MAINTENANCE: An Interconnect that is undergoing
    internal maintenance. No attachments may be provisioned or updated on this
    Interconnect.

    Values:
      OS_ACTIVE: The interconnect is valid, turned up, and ready to use.
        Attachments may be provisioned on this interconnect.
      OS_UNPROVISIONED: The interconnect has not completed turnup. No
        attachments may be provisioned on this interconnect.
    """
        OS_ACTIVE = 0
        OS_UNPROVISIONED = 1

    class RequestedFeaturesValueListEntryValuesEnum(_messages.Enum):
        """RequestedFeaturesValueListEntryValuesEnum enum type.

    Values:
      IF_MACSEC: Media Access Control security (MACsec)
    """
        IF_MACSEC = 0

    class StateValueValuesEnum(_messages.Enum):
        """[Output Only] The current state of Interconnect functionality, which
    can take one of the following values: - ACTIVE: The Interconnect is valid,
    turned up and ready to use. Attachments may be provisioned on this
    Interconnect. - UNPROVISIONED: The Interconnect has not completed turnup.
    No attachments may be provisioned on this Interconnect. -
    UNDER_MAINTENANCE: The Interconnect is undergoing internal maintenance. No
    attachments may be provisioned or updated on this Interconnect.

    Values:
      ACTIVE: The interconnect is valid, turned up, and ready to use.
        Attachments may be provisioned on this interconnect.
      UNPROVISIONED: The interconnect has not completed turnup. No attachments
        may be provisioned on this interconnect.
    """
        ACTIVE = 0
        UNPROVISIONED = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels for this resource. These can only be added or modified by the
    setLabels method. Each label key/value pair must comply with RFC1035.
    Label values may be empty.

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
    adminEnabled = _messages.BooleanField(1)
    availableFeatures = _messages.EnumField('AvailableFeaturesValueListEntryValuesEnum', 2, repeated=True)
    circuitInfos = _messages.MessageField('InterconnectCircuitInfo', 3, repeated=True)
    creationTimestamp = _messages.StringField(4)
    customerName = _messages.StringField(5)
    description = _messages.StringField(6)
    expectedOutages = _messages.MessageField('InterconnectOutageNotification', 7, repeated=True)
    googleIpAddress = _messages.StringField(8)
    googleReferenceId = _messages.StringField(9)
    id = _messages.IntegerField(10, variant=_messages.Variant.UINT64)
    interconnectAttachments = _messages.StringField(11, repeated=True)
    interconnectType = _messages.EnumField('InterconnectTypeValueValuesEnum', 12)
    kind = _messages.StringField(13, default='compute#interconnect')
    labelFingerprint = _messages.BytesField(14)
    labels = _messages.MessageField('LabelsValue', 15)
    linkType = _messages.EnumField('LinkTypeValueValuesEnum', 16)
    location = _messages.StringField(17)
    macsec = _messages.MessageField('InterconnectMacsec', 18)
    macsecEnabled = _messages.BooleanField(19)
    name = _messages.StringField(20)
    nocContactEmail = _messages.StringField(21)
    operationalStatus = _messages.EnumField('OperationalStatusValueValuesEnum', 22)
    peerIpAddress = _messages.StringField(23)
    provisionedLinkCount = _messages.IntegerField(24, variant=_messages.Variant.INT32)
    remoteLocation = _messages.StringField(25)
    requestedFeatures = _messages.EnumField('RequestedFeaturesValueListEntryValuesEnum', 26, repeated=True)
    requestedLinkCount = _messages.IntegerField(27, variant=_messages.Variant.INT32)
    satisfiesPzs = _messages.BooleanField(28)
    selfLink = _messages.StringField(29)
    state = _messages.EnumField('StateValueValuesEnum', 30)