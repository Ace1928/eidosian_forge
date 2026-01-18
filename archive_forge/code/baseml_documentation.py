import os
import os.path
from ._paml import Paml
from . import _parse_baseml
Run baseml using the current configuration.

        Check that the tree attribute is specified and exists, and then
        run baseml. If parse is True then read and return the result,
        otherwise return none.

        The arguments may be passed as either absolute or relative paths,
        despite the fact that BASEML requires relative paths.
        