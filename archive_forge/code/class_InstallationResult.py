import collections
import logging
from typing import Generator, List, Optional, Sequence, Tuple
from pip._internal.utils.logging import indent_log
from .req_file import parse_requirements
from .req_install import InstallRequirement
from .req_set import RequirementSet
class InstallationResult:

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f'InstallationResult(name={self.name!r})'