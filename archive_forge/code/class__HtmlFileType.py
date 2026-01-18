import os
from typing import TYPE_CHECKING, Sequence, Type, Union
from wandb.sdk.lib import filesystem, runid
from . import _dtypes
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia
class _HtmlFileType(_dtypes.Type):
    name = 'html-file'
    types = [Html]