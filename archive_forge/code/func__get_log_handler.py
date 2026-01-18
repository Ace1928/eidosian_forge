from __future__ import annotations
import logging
import typing as t
from copy import deepcopy
from textwrap import dedent
from traitlets.traitlets import (
from traitlets.utils import warnings
from traitlets.utils.bunch import Bunch
from traitlets.utils.text import indent, wrap_paragraphs
from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key
def _get_log_handler(self) -> logging.Handler | None:
    """Return the default Handler

        Returns None if none can be found

        Deprecated, this now returns the first log handler which may or may
        not be the default one.
        """
    if not self.log:
        return None
    logger: logging.Logger = self.log if isinstance(self.log, logging.Logger) else self.log.logger
    if not getattr(logger, 'handlers', None):
        return None
    return logger.handlers[0]