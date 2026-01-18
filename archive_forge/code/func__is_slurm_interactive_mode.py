import logging
import os
import re
import shutil
import signal
import sys
from typing import Optional
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.imports import _IS_WINDOWS
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.warnings import PossibleUserWarning
def _is_slurm_interactive_mode() -> bool:
    return SLURMEnvironment.job_name() in ('bash', 'interactive')