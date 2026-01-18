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
class LoggingConfigurable(Configurable):
    """A parent class for Configurables that log.

    Subclasses have a log trait, and the default behavior
    is to get the logger from the currently running Application.
    """
    log = Any(help='Logger or LoggerAdapter instance', allow_none=False)

    @validate('log')
    def _validate_log(self, proposal: Bunch) -> LoggerType:
        if not isinstance(proposal.value, (logging.Logger, logging.LoggerAdapter)):
            warnings.warn(f'{self.__class__.__name__}.log should be a Logger or LoggerAdapter, got {proposal.value}.', UserWarning, stacklevel=2)
        return t.cast(LoggerType, proposal.value)

    @default('log')
    def _log_default(self) -> LoggerType:
        if isinstance(self.parent, LoggingConfigurable):
            assert self.parent is not None
            return t.cast(logging.Logger, self.parent.log)
        from traitlets import log
        return log.get_logger()

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