import enum
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import os
import os.path
import sys
import traceback
class PycInvalidationMode(enum.Enum):
    TIMESTAMP = 1
    CHECKED_HASH = 2
    UNCHECKED_HASH = 3