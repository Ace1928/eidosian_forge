from __future__ import print_function, absolute_import
import sys
from shibokensupport.signature import inspect
from shibokensupport.signature import get_signature
class HintingEnumerator(ExactEnumerator):
    """
    HintingEnumerator enumerates all signatures in a module slightly changed.

    This class is used for generating complete listings of all signatures for
    hinting stubs. Only default values are replaced by "...".
    """

    def function(self, func_name, func, modifier=None):
        ret = self.result_type()
        signature = get_signature(func, 'hintingstub')
        if signature is not None:
            with self.fmt.function(func_name, signature, modifier) as key:
                ret[key] = signature
        return ret