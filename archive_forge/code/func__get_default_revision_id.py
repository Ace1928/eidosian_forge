import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
@functools.lru_cache(maxsize=1)
def _get_default_revision_id() -> Optional[str]:
    """Get the default revision ID based on `git describe`."""
    try:
        return exec_git(['describe', '--tags', '--dirty'])
    except BaseException:
        return None