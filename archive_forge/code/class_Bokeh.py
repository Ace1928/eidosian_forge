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
class Bokeh(Media):
    """Wandb class for Bokeh plots.

    Arguments:
        val: Bokeh plot
    """
    _log_type = 'bokeh-file'

    def __init__(self, data_or_path):
        super().__init__()
        bokeh = util.get_module('bokeh', required=True)
        if isinstance(data_or_path, str) and os.path.exists(data_or_path):
            with open(data_or_path) as file:
                b_json = json.load(file)
            self.b_obj = bokeh.document.Document.from_json(b_json)
            self._set_file(data_or_path, is_tmp=False, extension='.bokeh.json')
        elif isinstance(data_or_path, bokeh.model.Model):
            _data = bokeh.document.Document()
            _data.add_root(data_or_path)
            self.b_obj = bokeh.document.Document.from_json(_data.to_json())
            b_json = self.b_obj.to_json()
            if 'references' in b_json['roots']:
                b_json['roots']['references'].sort(key=lambda x: x['id'])
            tmp_path = os.path.join(MEDIA_TMP.name, runid.generate_id() + '.bokeh.json')
            with codecs.open(tmp_path, 'w', encoding='utf-8') as fp:
                util.json_dump_safer(b_json, fp)
            self._set_file(tmp_path, is_tmp=True, extension='.bokeh.json')
        elif not isinstance(data_or_path, bokeh.document.Document):
            raise TypeError('Bokeh constructor accepts Bokeh document/model or path to Bokeh json file')

    def get_media_subdir(self):
        return os.path.join('media', 'bokeh')

    def to_json(self, run):
        json_dict = super().to_json(run)
        json_dict['_type'] = self._log_type
        return json_dict

    @classmethod
    def from_json(cls, json_obj, source_artifact):
        return cls(source_artifact.get_entry(json_obj['path']).download())