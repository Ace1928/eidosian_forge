import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
def _try_tokenizer(mod_name):
    """Look for a tokenizer in the named module.

    Returns the function if found, None otherwise.
    """
    mod_base = 'enchant.tokenize.'
    func_name = 'tokenize'
    mod_name = mod_base + mod_name
    try:
        mod = __import__(mod_name, globals(), {}, func_name)
        return getattr(mod, func_name)
    except ImportError:
        return None