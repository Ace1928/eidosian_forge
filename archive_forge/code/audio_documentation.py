import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Union
import numpy as np
import pyarrow as pa
from .. import config
from ..download.download_config import DownloadConfig
from ..download.streaming_download_manager import xopen, xsplitext
from ..table import array_cast
from ..utils.py_utils import no_op_if_value_is_null, string_to_dict
Embed audio files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.

        Returns:
            `pa.StructArray`: Array in the Audio arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        