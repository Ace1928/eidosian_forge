from __future__ import print_function, absolute_import
import sys
from shibokensupport.signature import inspect
from shibokensupport.signature import get_signature
class SimplifyingEnumerator(ExactEnumerator):
    """
    SimplifyingEnumerator enumerates all signatures in a module filtered.

    There are no default values, no variable
    names and no self parameter. Only types are present after simplification.
    The functions 'next' resp. '__next__' are removed
    to make the output identical for Python 2 and 3.
    An appropriate formatter should be supplied, if printable output
    is desired.
    """

    def function(self, func_name, func, modifier=None):
        ret = self.result_type()
        signature = get_signature(func, 'existence')
        sig = stringify(signature) if signature is not None else None
        if sig is not None and func_name not in ('next', '__next__', '__div__'):
            with self.fmt.function(func_name, sig) as key:
                ret[key] = sig
        return ret