from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Union
from warnings import warn
from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.config import run_in_executor
from langchain.chains.base import Chain
@property
def _skip_input_warning(self) -> str:
    """Warning to show when input is ignored."""
    return f'Ignoring input in {self.__class__.__name__}, as it is not expected.'