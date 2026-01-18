import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple
import yaml
import ray  # noqa: F401
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_WORKER, TAG_RAY_NODE_KIND
import psutil
def _info_from_params(cluster: Optional[str]=None, host: Optional[str]=None, ssh_user: Optional[str]=None, ssh_key: Optional[str]=None, docker: Optional[str]=None):
    """Parse command line arguments.

    Note: This returns a list of hosts, not a comma separated string!
    """
    if not host and (not cluster):
        bootstrap_config = os.path.expanduser('~/ray_bootstrap_config.yaml')
        if os.path.exists(bootstrap_config):
            cluster = bootstrap_config
            cli_logger.warning(f'Detected cluster config file at {cluster}. If this is incorrect, specify with `ray cluster-dump <config>`')
    elif cluster:
        cluster = os.path.expanduser(cluster)
    cluster_name = None
    if cluster:
        h, u, k, d, cluster_name = get_info_from_ray_cluster_config(cluster)
        ssh_user = ssh_user or u
        ssh_key = ssh_key or k
        docker = docker or d
        hosts = host.split(',') if host else h
        if not hosts:
            raise LocalCommandFailed(f'Invalid cluster file or cluster has no running nodes: {cluster}')
    elif host:
        hosts = host.split(',')
    else:
        raise LocalCommandFailed('You need to either specify a `<cluster_config>` or `--host`.')
    if not ssh_user:
        ssh_user = DEFAULT_SSH_USER
        cli_logger.warning(f'Using default SSH user `{ssh_user}`. If this is incorrect, specify with `--ssh-user <user>`')
    if not ssh_key:
        for cand_key in DEFAULT_SSH_KEYS:
            cand_key_file = os.path.expanduser(cand_key)
            if os.path.exists(cand_key_file):
                ssh_key = cand_key_file
                cli_logger.warning(f'Auto detected SSH key file: {ssh_key}. If this is incorrect, specify with `--ssh-key <key>`')
                break
    return (cluster, hosts, ssh_user, ssh_key, docker, cluster_name)