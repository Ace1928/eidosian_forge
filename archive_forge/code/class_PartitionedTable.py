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
class PartitionedTable(Media):
    """A table which is composed of multiple sub-tables.

    Currently, PartitionedTable is designed to point to a directory within an artifact.
    """
    _log_type = 'partitioned-table'

    def __init__(self, parts_path):
        """Initialize a PartitionedTable.

        Args:
            parts_path (str): path to a directory of tables in the artifact.
        """
        super().__init__()
        self.parts_path = parts_path
        self._loaded_part_entries = {}

    def to_json(self, artifact_or_run):
        json_obj = {'_type': PartitionedTable._log_type}
        if isinstance(artifact_or_run, wandb.wandb_sdk.wandb_run.Run):
            artifact_entry_url = self._get_artifact_entry_ref_url()
            if artifact_entry_url is None:
                raise ValueError('PartitionedTables must first be added to an Artifact before logging to a Run')
            json_obj['artifact_path'] = artifact_entry_url
        else:
            json_obj['parts_path'] = self.parts_path
        return json_obj

    @classmethod
    def from_json(cls, json_obj, source_artifact):
        instance = cls(json_obj['parts_path'])
        entries = source_artifact.manifest.get_entries_in_directory(json_obj['parts_path'])
        for entry in entries:
            instance._add_part_entry(entry, source_artifact)
        return instance

    def iterrows(self):
        """Iterate over rows as (ndx, row).

        Yields:
        ------
        index : int
            The index of the row.
        row : List[any]
            The data of the row.
        """
        columns = None
        ndx = 0
        for entry_path in self._loaded_part_entries:
            part = self._loaded_part_entries[entry_path].get_part()
            if columns is None:
                columns = part.columns
            elif columns != part.columns:
                raise ValueError('Table parts have non-matching columns. {} != {}'.format(columns, part.columns))
            for _, row in part.iterrows():
                yield (ndx, row)
                ndx += 1
            self._loaded_part_entries[entry_path].free()

    def _add_part_entry(self, entry, source_artifact):
        self._loaded_part_entries[entry.path] = _PartitionTablePartEntry(entry, source_artifact)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.parts_path == other.parts_path

    def bind_to_run(self, *args, **kwargs):
        raise ValueError('PartitionedTables cannot be bound to runs')