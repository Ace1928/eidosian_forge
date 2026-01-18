from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectAttachmentConfigurationConstraints(_messages.Message):
    """A InterconnectAttachmentConfigurationConstraints object.

  Enums:
    BgpMd5ValueValuesEnum: [Output Only] Whether the attachment's BGP session
      requires/allows/disallows BGP MD5 authentication. This can take one of
      the following values: MD5_OPTIONAL, MD5_REQUIRED, MD5_UNSUPPORTED. For
      example, a Cross-Cloud Interconnect connection to a remote cloud
      provider that requires BGP MD5 authentication has the
      interconnectRemoteLocation attachment_configuration_constraints.bgp_md5
      field set to MD5_REQUIRED, and that property is propagated to the
      attachment. Similarly, if BGP MD5 is MD5_UNSUPPORTED, an error is
      returned if MD5 is requested.

  Fields:
    bgpMd5: [Output Only] Whether the attachment's BGP session
      requires/allows/disallows BGP MD5 authentication. This can take one of
      the following values: MD5_OPTIONAL, MD5_REQUIRED, MD5_UNSUPPORTED. For
      example, a Cross-Cloud Interconnect connection to a remote cloud
      provider that requires BGP MD5 authentication has the
      interconnectRemoteLocation attachment_configuration_constraints.bgp_md5
      field set to MD5_REQUIRED, and that property is propagated to the
      attachment. Similarly, if BGP MD5 is MD5_UNSUPPORTED, an error is
      returned if MD5 is requested.
    bgpPeerAsnRanges: [Output Only] List of ASN ranges that the remote
      location is known to support. Formatted as an array of inclusive ranges
      {min: min-value, max: max-value}. For example, [{min: 123, max: 123},
      {min: 64512, max: 65534}] allows the peer ASN to be 123 or anything in
      the range 64512-65534. This field is only advisory. Although the API
      accepts other ranges, these are the ranges that we recommend.
  """

    class BgpMd5ValueValuesEnum(_messages.Enum):
        """[Output Only] Whether the attachment's BGP session
    requires/allows/disallows BGP MD5 authentication. This can take one of the
    following values: MD5_OPTIONAL, MD5_REQUIRED, MD5_UNSUPPORTED. For
    example, a Cross-Cloud Interconnect connection to a remote cloud provider
    that requires BGP MD5 authentication has the interconnectRemoteLocation
    attachment_configuration_constraints.bgp_md5 field set to MD5_REQUIRED,
    and that property is propagated to the attachment. Similarly, if BGP MD5
    is MD5_UNSUPPORTED, an error is returned if MD5 is requested.

    Values:
      MD5_OPTIONAL: MD5_OPTIONAL: BGP MD5 authentication is supported and can
        optionally be configured.
      MD5_REQUIRED: MD5_REQUIRED: BGP MD5 authentication must be configured.
      MD5_UNSUPPORTED: MD5_UNSUPPORTED: BGP MD5 authentication must not be
        configured
    """
        MD5_OPTIONAL = 0
        MD5_REQUIRED = 1
        MD5_UNSUPPORTED = 2
    bgpMd5 = _messages.EnumField('BgpMd5ValueValuesEnum', 1)
    bgpPeerAsnRanges = _messages.MessageField('InterconnectAttachmentConfigurationConstraintsBgpPeerASNRange', 2, repeated=True)