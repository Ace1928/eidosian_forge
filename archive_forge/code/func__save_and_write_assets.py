import functools
import os
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _save_and_write_assets(self, assets_collection_to_add=None):
    """Saves asset to the meta graph and writes asset files to disk.

    Args:
      assets_collection_to_add: The collection where the asset paths are setup.
    """
    asset_filename_map = _maybe_save_assets(_add_asset_to_collection, assets_collection_to_add)
    if not asset_filename_map:
        tf_logging.info('No assets to write.')
        return
    copy_assets_to_destination_dir(asset_filename_map, self._export_dir, self._saved_asset_files)