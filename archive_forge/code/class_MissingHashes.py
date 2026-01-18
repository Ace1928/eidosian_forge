import hashlib
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional
from pip._internal.exceptions import HashMismatch, HashMissing, InstallationError
from pip._internal.utils.misc import read_chunks
class MissingHashes(Hashes):
    """A workalike for Hashes used when we're missing a hash for a requirement

    It computes the actual hash of the requirement and raises a HashMissing
    exception showing it to the user.

    """

    def __init__(self) -> None:
        """Don't offer the ``hashes`` kwarg."""
        super().__init__(hashes={FAVORITE_HASH: []})

    def _raise(self, gots: Dict[str, '_Hash']) -> 'NoReturn':
        raise HashMissing(gots[FAVORITE_HASH].hexdigest())