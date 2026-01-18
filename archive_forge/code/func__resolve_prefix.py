import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def _resolve_prefix(self, token):
    """Resolve command prefix from the prefix itself or its alias.

    Args:
      token: a str to be resolved.

    Returns:
      If resolvable, the resolved command prefix.
      If not resolvable, None.
    """
    if token in self._handlers:
        return token
    elif token in self._alias_to_prefix:
        return self._alias_to_prefix[token]
    else:
        return None