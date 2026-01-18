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
@tf_export(v1=['saved_model.Builder', 'saved_model.builder.SavedModelBuilder'])
class SavedModelBuilder(_SavedModelBuilder):
    __doc__ = _SavedModelBuilder.__doc__.replace('assets_list', 'assets_collection')

    def __init__(self, export_dir):
        super(SavedModelBuilder, self).__init__(export_dir=export_dir)

    def _add_collections(self, assets_collection, main_op, train_op):
        """Add asset and op collections to be saved."""
        self._save_and_write_assets(assets_collection)
        self._maybe_add_main_op(main_op)
        self._add_train_op(train_op)

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

    def _maybe_add_main_op(self, main_op):
        """Adds main op to the SavedModel.

    Args:
      main_op: Main op to run as part of graph initialization. If None, no main
        op will be added to the graph.

    Raises:
      TypeError: If the main op is provided but is not of type `Operation`.
      ValueError: if the Graph already contains an init op.
    """
        if main_op is None:
            return
        if not isinstance(main_op, ops.Operation):
            raise TypeError(f'Expected {main_op} to be an Operation but got type {type(main_op)} instead.')
        for init_op_key in (constants.MAIN_OP_KEY, constants.LEGACY_INIT_OP_KEY):
            if ops.get_collection(init_op_key):
                raise ValueError(f'Graph already contains one or more main ops under the collection {init_op_key}.')
        ops.add_to_collection(constants.MAIN_OP_KEY, main_op)

    def _add_train_op(self, train_op):
        """Add train op to the SavedModel.

    Note that this functionality is in development, and liable to be
    moved elsewhere.

    Args:
      train_op: Op or group of ops that are used for training. These are stored
        as a collection with key TRAIN_OP_KEY, but not executed.

    Raises:
      TypeError if Train op is not of type `Operation`.
    """
        if train_op is not None:
            if not isinstance(train_op, tensor.Tensor) and (not isinstance(train_op, ops.Operation)):
                raise TypeError(f'`train_op` {train_op} needs to be a Tensor or Op.')
            ops.add_to_collection(constants.TRAIN_OP_KEY, train_op)

    @deprecated_args(None, 'Pass your op to the equivalent parameter main_op instead.', 'legacy_init_op')
    def add_meta_graph(self, tags, signature_def_map=None, assets_collection=None, legacy_init_op=None, clear_devices=False, main_op=None, strip_default_attrs=False, saver=None):
        if not self._has_saved_variables:
            raise AssertionError('Graph state including variables and assets has not been saved yet. Please invoke `add_meta_graph_and_variables()` first.')
        signature_def_map = signature_def_map or {}
        self._validate_signature_def_map(signature_def_map)
        main_op = main_op if main_op is not None else legacy_init_op
        self._add_collections(assets_collection, main_op, None)
        saver = self._maybe_create_saver(saver)
        meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)
        self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

    @deprecated_args(None, 'Pass your op to the equivalent parameter main_op instead.', 'legacy_init_op')
    def add_meta_graph_and_variables(self, sess, tags, signature_def_map=None, assets_collection=None, legacy_init_op=None, clear_devices=False, main_op=None, strip_default_attrs=False, saver=None):
        if self._has_saved_variables:
            raise AssertionError('Graph state including variables and assets has already been saved. Please invoke `add_meta_graph()` instead.')
        signature_def_map = signature_def_map or {}
        self._validate_signature_def_map(signature_def_map)
        main_op = main_op or legacy_init_op
        self._add_collections(assets_collection, main_op, None)
        path_helpers.get_or_create_variables_dir(self._export_dir)
        variables_path = path_helpers.get_variables_path(self._export_dir)
        saver = self._maybe_create_saver(saver)
        saver.save(sess, variables_path, write_meta_graph=False, write_state=False)
        meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)
        self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)
        self._has_saved_variables = True
    add_meta_graph.__doc__ = _SavedModelBuilder.add_meta_graph.__doc__.replace('assets_list', 'assets_collection')
    add_meta_graph_and_variables.__doc__ = _SavedModelBuilder.add_meta_graph_and_variables.__doc__.replace('assets_list', 'assets_collection')