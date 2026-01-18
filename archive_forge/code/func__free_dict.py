import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def _free_dict(self, dict):
    """Free memory associated with a dictionary.

        This method frees system resources associated with a Dict object.
        It is equivalent to calling the object's 'free' method.  Once this
        method has been called on a dictionary, it must not be used again.
        """
    self._free_dict_data(dict._this)
    dict._this = None
    dict._broker = None