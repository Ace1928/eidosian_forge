import os
import warnings
class BiopythonDeprecationWarning(BiopythonWarning):
    """Biopython deprecation warning.

    Biopython uses this warning instead of the built in DeprecationWarning
    since those are ignored by default since Python 2.7.

    To silence all our deprecation warning messages, use:

    >>> import warnings
    >>> from Bio import BiopythonDeprecationWarning
    >>> warnings.simplefilter('ignore', BiopythonDeprecationWarning)

    Code marked as deprecated is likely to be removed in a future version
    of Biopython. To avoid removal of this code, please contact the Biopython
    developers via the mailing list or GitHub.
    """