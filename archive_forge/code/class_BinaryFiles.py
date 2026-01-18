import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
class BinaryFiles(BzrError):
    _fmt = 'Binary files section encountered.'

    def __init__(self, orig_name, mod_name):
        self.orig_name = orig_name
        self.mod_name = mod_name