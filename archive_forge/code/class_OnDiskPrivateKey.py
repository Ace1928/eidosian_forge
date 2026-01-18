from collections import namedtuple
from .agent import AgentKey
from .util import get_logger
from .ssh_exception import AuthenticationException
class OnDiskPrivateKey(PrivateKey):
    """
    Some on-disk private key that needs opening and possibly decrypting.

    :param str source:
        String tracking where this key's path was specified; should be one of
        ``"ssh-config"``, ``"python-config"``, or ``"implicit-home"``.
    :param Path path:
        The filesystem path this key was loaded from.
    :param PKey pkey:
        The `PKey` object this auth source uses/represents.
    """

    def __init__(self, username, source, path, pkey):
        super().__init__(username=username)
        self.source = source
        allowed = ('ssh-config', 'python-config', 'implicit-home')
        if source not in allowed:
            raise ValueError(f'source argument must be one of: {allowed!r}')
        self.path = path
        self.pkey = pkey

    def __repr__(self):
        return self._repr(key=self.pkey, source=self.source, path=str(self.path))