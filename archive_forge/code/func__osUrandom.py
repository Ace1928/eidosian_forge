import os
import random
import warnings
def _osUrandom(self, nbytes: int) -> bytes:
    """
        Wrapper around C{os.urandom} that cleanly manage its absence.
        """
    try:
        return os.urandom(nbytes)
    except (AttributeError, NotImplementedError) as e:
        raise SourceNotAvailable(e)