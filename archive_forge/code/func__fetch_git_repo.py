import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import click
import wandb
import wandb.docker as docker
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.wandb_reference import WandbReference
from wandb.sdk.wandb_config import Config
from .builder.templates._wandb_bootstrap import (
def _fetch_git_repo(dst_dir: str, uri: str, version: Optional[str]) -> Optional[str]:
    """Clones the git repo at ``uri`` into ``dst_dir``.

    checks out commit ``version``. Assumes authentication parameters are
    specified by the environment, e.g. by a Git credential helper.
    """
    _logger.info('Fetching git repo')
    ref = GitReference(uri, version)
    if ref is None:
        raise LaunchError(f'Unable to parse git uri: {uri}')
    ref.fetch(dst_dir)
    if version is None:
        version = ref.ref
    return version