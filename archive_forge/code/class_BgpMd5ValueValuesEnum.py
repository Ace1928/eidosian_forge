from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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