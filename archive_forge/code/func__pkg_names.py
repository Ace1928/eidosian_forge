import os
from distutils import log
import itertools
@staticmethod
def _pkg_names(pkg):
    """
        Given a namespace package, yield the components of that
        package.

        >>> names = Installer._pkg_names('a.b.c')
        >>> set(names) == set(['a', 'a.b', 'a.b.c'])
        True
        """
    parts = pkg.split('.')
    while parts:
        yield '.'.join(parts)
        parts.pop()