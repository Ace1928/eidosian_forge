from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuicOverrideValueValuesEnum(_messages.Enum):
    """Specifies the QUIC override policy for this TargetHttpsProxy resource.
    This setting determines whether the load balancer attempts to negotiate
    QUIC with clients. You can specify NONE, ENABLE, or DISABLE. - When quic-
    override is set to NONE, Google manages whether QUIC is used. - When quic-
    override is set to ENABLE, the load balancer uses QUIC when possible. -
    When quic-override is set to DISABLE, the load balancer doesn't use QUIC.
    - If the quic-override flag is not specified, NONE is implied.

    Values:
      DISABLE: The load balancer will not attempt to negotiate QUIC with
        clients.
      ENABLE: The load balancer will attempt to negotiate QUIC with clients.
      NONE: No overrides to the default QUIC policy. This option is implicit
        if no QUIC override has been specified in the request.
    """
    DISABLE = 0
    ENABLE = 1
    NONE = 2