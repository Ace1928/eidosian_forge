from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthenticationPolicy(_messages.Message):
    """[Deprecated] The authentication settings for the backend service. The
  authentication settings for the backend service.

  Enums:
    PrincipalBindingValueValuesEnum: Define whether peer or origin identity
      should be used for principal. Default value is USE_PEER. If peer (or
      origin) identity is not available, either because peer/origin
      authentication is not defined, or failed, principal will be left unset.
      In other words, binding rule does not affect the decision to accept or
      reject request. This field can be set to one of the following: USE_PEER:
      Principal will be set to the identity from peer authentication.
      USE_ORIGIN: Principal will be set to the identity from origin
      authentication.

  Fields:
    origins: List of authentication methods that can be used for origin
      authentication. Similar to peers, these will be evaluated in order the
      first valid one will be used to set origin identity. If none of these
      methods pass, the request will be rejected with authentication failed
      error (401). Leave the list empty if origin authentication is not
      required.
    peers: List of authentication methods that can be used for peer
      authentication. They will be evaluated in order the first valid one will
      be used to set peer identity. If none of these methods pass, the request
      will be rejected with authentication failed error (401). Leave the list
      empty if peer authentication is not required.
    principalBinding: Define whether peer or origin identity should be used
      for principal. Default value is USE_PEER. If peer (or origin) identity
      is not available, either because peer/origin authentication is not
      defined, or failed, principal will be left unset. In other words,
      binding rule does not affect the decision to accept or reject request.
      This field can be set to one of the following: USE_PEER: Principal will
      be set to the identity from peer authentication. USE_ORIGIN: Principal
      will be set to the identity from origin authentication.
    serverTlsContext: Configures the mechanism to obtain server-side security
      certificates and identity information.
  """

    class PrincipalBindingValueValuesEnum(_messages.Enum):
        """Define whether peer or origin identity should be used for principal.
    Default value is USE_PEER. If peer (or origin) identity is not available,
    either because peer/origin authentication is not defined, or failed,
    principal will be left unset. In other words, binding rule does not affect
    the decision to accept or reject request. This field can be set to one of
    the following: USE_PEER: Principal will be set to the identity from peer
    authentication. USE_ORIGIN: Principal will be set to the identity from
    origin authentication.

    Values:
      INVALID: <no description>
      USE_ORIGIN: Principal will be set to the identity from origin
        authentication.
      USE_PEER: Principal will be set to the identity from peer
        authentication.
    """
        INVALID = 0
        USE_ORIGIN = 1
        USE_PEER = 2
    origins = _messages.MessageField('OriginAuthenticationMethod', 1, repeated=True)
    peers = _messages.MessageField('PeerAuthenticationMethod', 2, repeated=True)
    principalBinding = _messages.EnumField('PrincipalBindingValueValuesEnum', 3)
    serverTlsContext = _messages.MessageField('TlsContext', 4)