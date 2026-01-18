from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union
import torch
import torio
Context manager so that the destination is closed and data are flushed automatically.