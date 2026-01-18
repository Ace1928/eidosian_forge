import logging
import os
import re
from typing import TYPE_CHECKING, Optional, Sequence, Union, cast
import wandb
from wandb import util
from .base_types.media import BatchableMedia, Media
from .base_types.wb_value import WBValue
from .image import _server_accepts_image_filenames
from .plotly import Plotly
def _prune_max_seq(seq: Sequence['BatchableMedia']) -> Sequence['BatchableMedia']:
    items = seq
    if hasattr(seq[0], 'MAX_ITEMS') and seq[0].MAX_ITEMS < len(seq):
        logging.warning('Only %i %s will be uploaded.' % (seq[0].MAX_ITEMS, seq[0].__class__.__name__))
        items = seq[:seq[0].MAX_ITEMS]
    return items