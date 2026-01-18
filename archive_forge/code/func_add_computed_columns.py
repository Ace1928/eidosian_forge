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
def add_computed_columns(self, fn):
    """Adds one or more computed columns based on existing data.

        Args:
            fn: A function which accepts one or two parameters, ndx (int) and row (dict),
                which is expected to return a dict representing new columns for that row, keyed
                by the new column names.

                `ndx` is an integer representing the index of the row. Only included if `include_ndx`
                      is set to `True`.

                `row` is a dictionary keyed by existing columns
        """
    new_columns = {}
    for ndx, row in self.iterrows():
        row_dict = {self.columns[i]: row[i] for i in range(len(self.columns))}
        new_row_dict = fn(ndx, row_dict)
        assert isinstance(new_row_dict, dict)
        for key in new_row_dict:
            new_columns[key] = new_columns.get(key, [])
            new_columns[key].append(new_row_dict[key])
    for new_col_name in new_columns:
        self.add_column(new_col_name, new_columns[new_col_name])