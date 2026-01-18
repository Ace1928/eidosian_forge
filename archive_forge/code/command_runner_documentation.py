import os
import subprocess
from typing import Dict, List, Tuple
from ray.autoscaler._private.docker import with_docker_exec
from ray.autoscaler.command_runner import CommandRunnerInterface
Command runner for the fke docker multinode cluster.

    This command runner uses ``docker exec`` and ``docker cp`` to
    run commands and copy files, respectively.

    The regular ``DockerCommandRunner`` is made for use in SSH settings
    where Docker runs on a remote hose. In contrast, this command runner
    does not wrap the docker commands in ssh calls.
    