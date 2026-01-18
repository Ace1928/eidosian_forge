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
@staticmethod
def _validate_srun_used() -> None:
    """Checks if the `srun` command is available and used.

        Parallel jobs (multi-GPU, multi-node) in SLURM are launched by prepending `srun` in front of the Python command.
        Not doing so will result in processes hanging, which is a frequent user error. Lightning will emit a warning if
        `srun` is found but not used.

        """
    if _IS_WINDOWS:
        return
    srun_exists = shutil.which('srun') is not None
    if srun_exists and (not _is_srun_used()):
        hint = ' '.join(['srun', os.path.basename(sys.executable), *sys.argv])[:64]
        rank_zero_warn(f'The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: {hint} ...', category=PossibleUserWarning)