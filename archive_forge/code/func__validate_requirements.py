import collections
import logging
from typing import Generator, List, Optional, Sequence, Tuple
from pip._internal.utils.logging import indent_log
from .req_file import parse_requirements
from .req_install import InstallRequirement
from .req_set import RequirementSet
def _validate_requirements(requirements: List[InstallRequirement]) -> Generator[Tuple[str, InstallRequirement], None, None]:
    for req in requirements:
        assert req.name, f'invalid to-be-installed requirement: {req}'
        yield (req.name, req)