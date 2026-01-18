import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
@functools.lru_cache(maxsize=1)
def _get_compose_command() -> Optional[List[str]]:
    try:
        compose_command = get_docker_compose_command()
    except ValueError as e:
        compose_command = [f'NOT INSTALLED: {e}']
    except:
        return None
    return compose_command