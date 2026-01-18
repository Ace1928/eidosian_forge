import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.load import _METADATA_FILENAME, _load_distributed_checkpoint
def _parse_cli_args() -> Namespace:
    parser = ArgumentParser(description='Converts a distributed/sharded checkpoint into a single file that can be loaded with `torch.load()`. Only supports FSDP sharded checkpoints at the moment.')
    parser.add_argument('checkpoint_folder', type=str, help='Path to a checkpoint folder, containing the sharded checkpoint files saved using the `torch.distributed.checkpoint` API.')
    parser.add_argument('--output_file', type=str, help="Path to the file where the converted checkpoint should be saved. The file should not already exist. If no path is provided, the file will be saved next to the input checkpoint folder with the same name and a '.consolidated' suffix.")
    return parser.parse_args()