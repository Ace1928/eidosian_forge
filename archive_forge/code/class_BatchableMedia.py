import hashlib
import os
import platform
import re
import shutil
from typing import TYPE_CHECKING, Optional, Sequence, Type, Union, cast
import wandb
from wandb import util
from wandb._globals import _datatypes_callback
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.paths import LogicalPath
from .wb_value import WBValue
class BatchableMedia(Media):
    """Media that is treated in batches.

    E.g. images and thumbnails. Apart from images, we just use these batches to help
    organize files by name in the media directory.
    """

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def seq_to_json(cls: Type['BatchableMedia'], seq: Sequence['BatchableMedia'], run: 'LocalRun', key: str, step: Union[int, str]) -> dict:
        raise NotImplementedError