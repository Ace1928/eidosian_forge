from __future__ import annotations
import json
import os
import re
import shutil
import typing as t
import warnings
from jupyter_core.paths import SYSTEM_JUPYTER_PATH, jupyter_data_dir, jupyter_path
from traitlets import Bool, CaselessStrEnum, Dict, HasTraits, List, Set, Type, Unicode, observe
from traitlets.config import LoggingConfigurable
from .provisioning import KernelProvisionerFactory as KPF  # noqa
class NoSuchKernel(KeyError):
    """An error raised when there is no kernel of a give name."""

    def __init__(self, name: str) -> None:
        """Initialize the error."""
        self.name = name

    def __str__(self) -> str:
        return f'No such kernel named {self.name}'