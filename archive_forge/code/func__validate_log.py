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
@validate('log')
def _validate_log(self, proposal: Bunch) -> LoggerType:
    if not isinstance(proposal.value, (logging.Logger, logging.LoggerAdapter)):
        warnings.warn(f'{self.__class__.__name__}.log should be a Logger or LoggerAdapter, got {proposal.value}.', UserWarning, stacklevel=2)
    return t.cast(LoggerType, proposal.value)