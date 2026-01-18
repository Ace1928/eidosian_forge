import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
class LoggingHasTraits(HasTraits):
    """A parent class for HasTraits that log.
    Subclasses have a log trait, and the default behavior
    is to get the logger from the currently running Application.
    """
    log = Instance('logging.Logger')

    @default('log')
    def _log_default(self):
        from traitlets import log
        return log.get_logger()