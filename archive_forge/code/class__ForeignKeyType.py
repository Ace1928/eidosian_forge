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
class _ForeignKeyType(_dtypes.Type):
    name = 'foreignKey'
    legacy_names = ['wandb.TableForeignKey']
    types = [_TableKey]

    def __init__(self, table, col_name):
        assert isinstance(table, Table)
        assert isinstance(col_name, str)
        assert col_name in table.columns
        self.params.update({'table': table, 'col_name': col_name})

    def assign_type(self, wb_type=None):
        if isinstance(wb_type, _dtypes.StringType):
            return self
        elif isinstance(wb_type, _ForeignKeyType) and id(self.params['table']) == id(wb_type.params['table']) and (self.params['col_name'] == wb_type.params['col_name']):
            return self
        return _dtypes.InvalidType()

    @classmethod
    def from_obj(cls, py_obj):
        if not isinstance(py_obj, _TableKey):
            raise TypeError('py_obj must be a _TableKey')
        else:
            return cls(py_obj._table, py_obj._col_name)

    def to_json(self, artifact=None):
        res = super().to_json(artifact)
        if artifact is not None:
            table_name = f'media/tables/t_{runid.generate_id()}'
            entry = artifact.add(self.params['table'], table_name)
            res['params']['table'] = entry.path
        else:
            raise AssertionError('_ForeignKeyType does not support serialization without an artifact')
        return res

    @classmethod
    def from_json(cls, json_dict, artifact):
        table = None
        col_name = None
        if artifact is None:
            raise AssertionError('_ForeignKeyType does not support deserialization without an artifact')
        else:
            table = artifact.get(json_dict['params']['table'])
            col_name = json_dict['params']['col_name']
        if table is None:
            raise AssertionError('Unable to deserialize referenced table')
        return cls(table, col_name)