import argparse
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import yaml
Fake multinode docker monitoring script.

This script is the "docker compose server" for the fake_multinode
provider using Docker compose. It should be started before running
`RAY_FAKE_CLUSTER=1 ray up <cluster_config>`.

This script reads the volume directory from a supplied fake multinode
docker cluster config file.
It then waits until a docker-compose.yaml file is created in the same
directory, which is done by the `ray up` command.

It then watches for changes in the docker-compose.yaml file and runs
`docker compose up` whenever changes are detected. This will start docker
containers as requested by the autoscaler.

Generally, the docker-compose.yaml will be mounted in the head node of the
cluster, which will then continue to change it according to the autoscaler
requirements.

Additionally, this script monitors the docker container status using
`docker status` and writes it into a `status.json`. This information is
again used by the autoscaler to determine if any nodes have died.
