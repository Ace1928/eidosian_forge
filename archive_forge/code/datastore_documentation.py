import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
Write a protocol buffer.

        Arguments:
            obj: Protocol buffer to write.

        Returns:
            (start_offset, end_offset, flush_offset) if successful,
            None otherwise

        