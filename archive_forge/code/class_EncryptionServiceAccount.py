import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class EncryptionServiceAccount(Reference):
    _required_fields = frozenset(('serviceAccount',))
    _format_str = '%(serviceAccount)s'
    typename = None