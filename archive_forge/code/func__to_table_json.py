import base64
import binascii
import codecs
import datetime
import hashlib
import json
import logging
import os
import pprint
from decimal import Decimal
from typing import Optional
import wandb
from wandb import util
from wandb.sdk.lib import filesystem
from .sdk.data_types import _dtypes
from .sdk.data_types._private import MEDIA_TMP
from .sdk.data_types.base_types.media import (
from .sdk.data_types.base_types.wb_value import WBValue
from .sdk.data_types.helper_types.bounding_boxes_2d import BoundingBoxes2D
from .sdk.data_types.helper_types.classes import Classes
from .sdk.data_types.helper_types.image_mask import ImageMask
from .sdk.data_types.histogram import Histogram
from .sdk.data_types.html import Html
from .sdk.data_types.image import Image
from .sdk.data_types.molecule import Molecule
from .sdk.data_types.object_3d import Object3D
from .sdk.data_types.plotly import Plotly
from .sdk.data_types.saved_model import _SavedModel
from .sdk.data_types.trace_tree import WBTraceTree
from .sdk.data_types.video import Video
from .sdk.lib import runid
def _to_table_json(self, max_rows=None, warn=True):
    if max_rows is None:
        max_rows = Table.MAX_ROWS
    n_rows = len(self.data)
    if n_rows > max_rows and warn:
        if wandb.run and (wandb.run.settings.table_raise_on_max_row_limit_exceeded or wandb.run.settings.strict):
            raise ValueError(f'Table row limit exceeded: table has {n_rows} rows, limit is {max_rows}. To increase the maximum number of allowed rows in a wandb.Table, override the limit with `wandb.Table.MAX_ARTIFACT_ROWS = X` and try again. Note: this may cause slower queries in the W&B UI.')
        logging.warning('Truncating wandb.Table object to %i rows.' % max_rows)
    return {'columns': self.columns, 'data': self.data[:max_rows]}