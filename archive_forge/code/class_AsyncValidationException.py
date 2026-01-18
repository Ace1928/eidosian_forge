import asyncio
import json
import pathlib
import re
import logging
from typing import (
import inspect
from inspect import signature, iscoroutinefunction
from collections.abc import Mapping, Iterable
from enum import Enum
import importlib
import os
import aiofiles
from regex import W
import asyncio
import types
import importlib.util
class AsyncValidationException(ValueError):
    """
    A specific exception type for async validation failures, providing detailed information about the failed validation.

    Attributes:
        argument (str): The name of the argument that failed validation.
        value (Any): The value of the argument that failed validation.
        message (str): An optional more detailed message.
    """

    def __init__(self, argument: str, value: Any, message: str='') -> None:
        self.argument = argument
        self.value = value
        super().__init__(message or f"Validation failed for argument '{argument}' with value '{value}'")

    def __str__(self) -> str:
        return f"Validation failed for argument '{self.argument}' with value '{self.value}'"