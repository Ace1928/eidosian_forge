from __future__ import annotations
import logging # isort:skip
import os
import sys
import traceback
from os.path import basename
from types import CodeType, ModuleType
from typing import Callable
from ...core.types import PathLike
from ...util.serialization import make_globally_unique_id
from .handler import handle_exception
 Execute the configured source code in a module and run any post
        checks.

        Args:
            module (Module) :
                A module to execute the configured code in.

            post_check (callable, optional) :
                A function that raises an exception if expected post-conditions
                are not met after code execution.

        