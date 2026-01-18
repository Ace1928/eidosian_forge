from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union
import torch
import torio
def _print_output_stream(self, i: int):
    """[debug] Print the registered stream information to stdout."""
    self._s.dump_format(i)