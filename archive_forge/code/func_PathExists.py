import os
import tarfile
from ._basic import Equals
from ._higherorder import (
from ._impl import (
def PathExists():
    """Matches if the given path exists.

    Use like this::

      assertThat('/some/path', PathExists())
    """
    return MatchesPredicate(os.path.exists, '%s does not exist.')