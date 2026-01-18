import dataclasses
import typing
from spnego._ntlm_raw.crypto import is_ntlm_hash
from spnego.exceptions import InvalidCredentialError, NoCredentialError
@dataclasses.dataclass
class CredentialCache:
    """Cached credential.

    Uses the provider specific cached credential. Can also specify a username
    to select a specific cached credential.

    Attributes:
        username: Optional username used to select a specific credential in the
            cache.
    """
    username: typing.Optional[str] = None

    @property
    def supported_protocols(self) -> typing.List[str]:
        return ['kerberos', 'ntlm']