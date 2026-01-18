import re
from typing import Any, Dict
import torch
from lightning_fabric.utilities.consolidate_checkpoint import _parse_cli_args, _process_cli_args
from lightning_fabric.utilities.load import _load_distributed_checkpoint
def _format_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Converts the special FSDP checkpoint format to the standard format the Lightning Trainer can load."""
    checkpoint['state_dict'] = checkpoint.pop('model')
    optimizer_keys = [key for key in checkpoint if re.match('optimizer_[0-9]+', key)]
    if not optimizer_keys:
        return checkpoint
    checkpoint['optimizer_states'] = [checkpoint.pop(f'optimizer_{opt_idx}') for opt_idx in range(len(optimizer_keys))]
    return checkpoint