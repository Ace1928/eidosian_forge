import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
def _set_offset(self, offset):
    msg = "changing a tokenizers 'offset' attribute is deprecated; use the 'set_offset' method"
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
    self.set_offset(offset)