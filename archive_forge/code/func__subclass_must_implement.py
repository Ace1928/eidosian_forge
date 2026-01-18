from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def _subclass_must_implement(self, fn):
    """
        Returns a NotImplementedError for a function that should be implemented.
        :param fn: name of the function
        """
    m = 'Missing function implementation in {}: {}'.format(type(self), fn)
    return NotImplementedError(m)