import dataclasses
import typing
from spnego._ntlm_raw.crypto import is_ntlm_hash
from spnego.exceptions import InvalidCredentialError, NoCredentialError
@dataclasses.dataclass
class KerberosCCache:
    """Kerberos CCache Credential.

    Used with :class:`GSSAPIProxy` for Kerberos authentication. It is used to
    specify the credential cache that has the stored Kerberos credential for
    authentication. The ccache value is specified in the form ``TYPE:RESIDUAL``
    where the ``TYPE`` supported is down to the installed Kerberos/GSSAPI
    implementation and ``RESIDUAL`` is a value specific to the type. Common
    types are:

        DIR: The value is the path to a directory containing a collection of
            `FILE` caches.
        FILE: The value is the path to an individual cache.
        MEMORY: The value is a unique identifier to a cache stored in memory of
            the current process. It must be resolvable by the linked GSSAPI
            provider that this library uses.

    There are other ccache types but they are mostly platform or GSSAPI
    implementation specific.

    .. Note:
        This only works on Linux, Windows does not have the concept of
        separate CCaches.

    Attributes:
        ccache: The ccache in the form ``TYPE:RESIDUAL`` to use for a Kerberos
            credential. The path will not be expanded of have variables
            substituted so should be the absolute path to the ccache.
        principal: Optional principal to get in the credential cache specified.
    """
    ccache: str
    principal: typing.Optional[str] = None

    @property
    def supported_protocols(self) -> typing.List[str]:
        """List of protocols the credential can be used for."""
        return ['kerberos']