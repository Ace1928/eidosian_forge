import dataclasses
import typing
from spnego._ntlm_raw.crypto import is_ntlm_hash
from spnego.exceptions import InvalidCredentialError, NoCredentialError
@dataclasses.dataclass
class NTLMHash:
    """NTLM LM/NT Hash credential.

    Used with :class:`NTLMProxy` for NTLM authentication backed by an NT and/or
    LM hash value. In modern iterations of NTLM only the NT hash needs to be
    specified and LM is ignored but both can be specified and the NTLM code
    will use them as necessary.

    Attributes:
        username: The username the hashes are for.
        lm_hash: The LM hash as a hex string, can be `None` in most cases.
        nt_hash: The NT hash as a hex string.
    """
    username: str
    lm_hash: typing.Optional[str] = dataclasses.field(default=None, repr=False)
    nt_hash: typing.Optional[str] = dataclasses.field(default=None, repr=False)

    @property
    def supported_protocols(self) -> typing.List[str]:
        """List of protocols the credential can be used for."""
        return ['ntlm']