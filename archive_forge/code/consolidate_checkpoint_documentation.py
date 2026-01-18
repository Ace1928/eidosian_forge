import re
from typing import Any, Dict
import torch
from lightning_fabric.utilities.consolidate_checkpoint import _parse_cli_args, _process_cli_args
from lightning_fabric.utilities.load import _load_distributed_checkpoint
Converts the special FSDP checkpoint format to the standard format the Lightning Trainer can load.