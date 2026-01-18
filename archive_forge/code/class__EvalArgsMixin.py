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
class _EvalArgsMixin:
    """Mixin for checking evaluation arguments."""

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return False

    @property
    def requires_input(self) -> bool:
        """Whether this evaluator requires an input string."""
        return False

    @property
    def _skip_input_warning(self) -> str:
        """Warning to show when input is ignored."""
        return f'Ignoring input in {self.__class__.__name__}, as it is not expected.'

    @property
    def _skip_reference_warning(self) -> str:
        """Warning to show when reference is ignored."""
        return f'Ignoring reference in {self.__class__.__name__}, as it is not expected.'

    def _check_evaluation_args(self, reference: Optional[str]=None, input: Optional[str]=None) -> None:
        """Check if the evaluation arguments are valid.

        Args:
            reference (Optional[str], optional): The reference label.
            input (Optional[str], optional): The input string.
        Raises:
            ValueError: If the evaluator requires an input string but none is provided,
                or if the evaluator requires a reference label but none is provided.
        """
        if self.requires_input and input is None:
            raise ValueError(f'{self.__class__.__name__} requires an input string.')
        elif input is not None and (not self.requires_input):
            warn(self._skip_input_warning)
        if self.requires_reference and reference is None:
            raise ValueError(f'{self.__class__.__name__} requires a reference string.')
        elif reference is not None and (not self.requires_reference):
            warn(self._skip_reference_warning)