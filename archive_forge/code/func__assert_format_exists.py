import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
@staticmethod
def _assert_format_exists(file_format, methods):
    """Checks whether the given file format exists in 'methods'.
        """
    try:
        return methods[file_format]
    except KeyError:
        formats = ', '.join(sorted(methods.keys()))
        raise ValueError('Unsupported format: %r, try one of %s' % (file_format, formats))