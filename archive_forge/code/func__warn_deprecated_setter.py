import re
import string
def _warn_deprecated_setter(setter):
    import warnings
    msg = 'The .%s setter is deprecated. The attribute will be read-only in future releases. Please use the set() method instead.' % setter
    warnings.warn(msg, DeprecationWarning, stacklevel=3)