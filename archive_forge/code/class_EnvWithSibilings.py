import importlib
import os
import secrets
import sys
import warnings
from textwrap import dedent
from typing import Any, Optional
from packaging import version
from pandas.util._decorators import doc  # type: ignore[attr-defined]
from modin.config.pubsub import (
class EnvWithSibilings(EnvironmentVariable, type=str):
    """Ensure values synchronization between sibling parameters."""
    _update_sibling = True

    @classmethod
    def _sibling(cls) -> type['EnvWithSibilings']:
        """Return a sibling parameter."""
        raise NotImplementedError()

    @classmethod
    def get(cls) -> Any:
        """
        Get parameter's value and ensure that it's equal to the sibling's value.

        Returns
        -------
        Any
        """
        sibling = cls._sibling()
        if sibling._value is _UNSET and cls._value is _UNSET:
            super().get()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                super(EnvWithSibilings, sibling).get()
            if cls._value_source == sibling._value_source == ValueSource.GOT_FROM_CFG_SOURCE:
                raise ValueError(f"Configuration is ambiguous. You cannot set '{cls.varname}' and '{sibling.varname}' at the same time.")
            from modin.error_message import ErrorMessage
            if cls._value_source == ValueSource.GOT_FROM_CFG_SOURCE:
                ErrorMessage.catch_bugs_and_request_email(failure_condition=sibling._value_source != ValueSource.DEFAULT)
                sibling._value = cls._value
                sibling._value_source = ValueSource.GOT_FROM_CFG_SOURCE
            elif sibling._value_source == ValueSource.GOT_FROM_CFG_SOURCE:
                ErrorMessage.catch_bugs_and_request_email(failure_condition=cls._value_source != ValueSource.DEFAULT)
                cls._value = sibling._value
                cls._value_source = ValueSource.GOT_FROM_CFG_SOURCE
            else:
                ErrorMessage.catch_bugs_and_request_email(failure_condition=cls._value_source != ValueSource.DEFAULT or sibling._value_source != ValueSource.DEFAULT)
                sibling._value = cls._value
        return super().get()

    @classmethod
    def put(cls, value: Any) -> None:
        """
        Set a new value to this parameter as well as to its sibling.

        Parameters
        ----------
        value : Any
        """
        super().put(value)
        if cls._update_sibling:
            cls._update_sibling = False
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=FutureWarning)
                    cls._sibling().put(value)
            finally:
                cls._update_sibling = True