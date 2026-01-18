import logging
import os
import time
from datetime import datetime
from shlex import quote, split
from threading import RLock
import kubernetes
from kubernetes.config.config_exception import ConfigException
import docker
from mlflow.entities import RunStatus
from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
def _get_run_command(entrypoint_command):
    formatted_command = []
    for cmd in entrypoint_command:
        formatted_command.extend([quote(s) for s in split(cmd)])
    return formatted_command