import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def _setinitialized() -> None:
    """Set the embedded R as initialized.

    This may result in a later segfault if used with the embedded R has not
    been initialized. You should not have to use it."""
    global rpy2_embeddedR_isinitialized
    rpy2_embeddedR_isinitialized = RPY_R_Status.INITIALIZED.value