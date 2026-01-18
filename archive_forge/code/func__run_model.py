import logging
import os
import re
import subprocess
import sys
from argparse import Namespace
from typing import Any, List, Optional
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import get_args
from lightning_fabric.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning_fabric.plugins.precision.precision import _PRECISION_INPUT_STR, _PRECISION_INPUT_STR_ALIAS
from lightning_fabric.strategies import STRATEGY_REGISTRY
from lightning_fabric.utilities.device_parser import _parse_gpu_ids
from lightning_fabric.utilities.distributed import _suggested_max_num_threads
@run.command('model', context_settings={'ignore_unknown_options': True})
@click.argument('script', type=click.Path(exists=True))
@click.option('--accelerator', type=click.Choice(_SUPPORTED_ACCELERATORS), default=None, help='The hardware accelerator to run on.')
@click.option('--strategy', type=click.Choice(_get_supported_strategies()), default=None, help='Strategy for how to run across multiple devices.')
@click.option('--devices', type=str, default='1', help="Number of devices to run on (``int``), which devices to run on (``list`` or ``str``), or ``'auto'``. The value applies per node.")
@click.option('--num-nodes', '--num_nodes', type=int, default=1, help='Number of machines (nodes) for distributed execution.')
@click.option('--node-rank', '--node_rank', type=int, default=0, help='The index of the machine (node) this command gets started on. Must be a number in the range 0, ..., num_nodes - 1.')
@click.option('--main-address', '--main_address', type=str, default='127.0.0.1', help='The hostname or IP address of the main machine (usually the one with node_rank = 0).')
@click.option('--main-port', '--main_port', type=int, default=29400, help='The main port to connect to the main machine.')
@click.option('--precision', type=click.Choice(get_args(_PRECISION_INPUT_STR) + get_args(_PRECISION_INPUT_STR_ALIAS)), default=None, help='Double precision (``64-true`` or ``64``), full precision (``32-true`` or ``64``), half precision (``16-mixed`` or ``16``) or bfloat16 precision (``bf16-mixed`` or ``bf16``)')
@click.argument('script_args', nargs=-1, type=click.UNPROCESSED)
def _run_model(**kwargs: Any) -> None:
    """Run a Lightning Fabric script.

        SCRIPT is the path to the Python script with the code to run. The script must contain a Fabric object.

        SCRIPT_ARGS are the remaining arguments that you can pass to the script itself and are expected to be parsed
        there.

        """
    script_args = list(kwargs.pop('script_args', []))
    main(args=Namespace(**kwargs), script_args=script_args)