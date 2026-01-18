import os
import subprocess
from typing import Dict, List, Tuple
from ray.autoscaler._private.docker import with_docker_exec
from ray.autoscaler.command_runner import CommandRunnerInterface
def _run_shell(self, cmd: str, timeout: int=120) -> str:
    return subprocess.check_output(cmd, shell=True, timeout=timeout, encoding='utf-8')