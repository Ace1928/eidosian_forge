from collections import namedtuple
from .agent import AgentKey
from .util import get_logger
from .ssh_exception import AuthenticationException
class NoneAuth(AuthSource):
    """
    Auth type "none", ie https://www.rfc-editor.org/rfc/rfc4252#section-5.2 .
    """

    def authenticate(self, transport):
        return transport.auth_none(self.username)