import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
class RPY_R_Status(enum.Enum):
    """Possible status for the embedded R."""
    INITIALIZED = 1
    BUSY = 2
    ENDED = 4