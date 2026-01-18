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
@tf_export('__internal__.saved_model.SavedModelBuilder', v1=[])
class _SavedModelBuilder(object):
    """Builds the `SavedModel` protocol buffer and saves variables and assets.

  The `SavedModelBuilder` class provides the functionality to build a
  `SavedModel` protocol buffer. Specifically, this allows multiple meta
  graphs to be saved as part of a single language-neutral `SavedModel`,
  while sharing variables and assets.

  To build a SavedModel, the first meta graph must be saved with variables.
  Subsequent meta graphs will simply be saved with their graph definitions. If
  assets need to be saved and written or copied to disk, they can be provided
  when the meta graph def is added. If multiple meta graph defs are associated
  an asset of the same name, only the first version is retained.

  Each meta graph added to the SavedModel must be annotated with tags. The tags
  provide a means to identify the specific meta graph to load and restore, along
  with the shared set of variables and assets.

  Typical usage for the `SavedModelBuilder`:

  ```python
  ...
  builder = tf.compat.v1.saved_model.Builder(export_dir)

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph_and_variables(sess,
                                    ["foo-tag"],
                                    signature_def_map=foo_signatures,
                                    assets_list=foo_assets)
  ...

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph(["bar-tag", "baz-tag"])
  ...

  builder.save()
  ```

  Note: This function will only be available through the v1 compatibility
  library as tf.compat.v1.saved_model.builder.SavedModelBuilder or
  tf.compat.v1.saved_model.Builder. Tensorflow 2.0 will introduce a new
  object-based method of creating SavedModels.
  """

    def __init__(self, export_dir):
        self._saved_model = saved_model_pb2.SavedModel()
        self._saved_model.saved_model_schema_version = constants.SAVED_MODEL_SCHEMA_VERSION
        self._export_dir = export_dir
        if file_io.file_exists(export_dir):
            if file_io.list_directory(export_dir):
                raise AssertionError(f"Export directory {export_dir} already exists, and isn't empty. Please choose a different export directory, or delete all the contents of the specified directory.")
        else:
            file_io.recursive_create_dir(self._export_dir)
        self._has_saved_variables = False
        self._saved_asset_files = set()

    def _save_and_write_assets(self, meta_graph_def, assets_list=None):
        """Saves asset to the meta graph and writes asset files to disk.

    Args:
      meta_graph_def: The meta graph def to which the assets will be added.
      assets_list: The list where the asset paths are setup.
    """
        write_fn = functools.partial(_add_asset_to_metagraph, meta_graph_def)
        asset_filename_map = _maybe_save_assets(write_fn, assets_list)
        if not asset_filename_map:
            tf_logging.info('No assets to write.')
            return
        copy_assets_to_destination_dir(asset_filename_map, self._export_dir, self._saved_asset_files)

    def _tag_and_add_meta_graph(self, meta_graph_def, tags, signature_def_map):
        """Tags the meta graph def and adds it to the SavedModel.

    Tags the meta graph def with the supplied tags, adds signature defs to it if
    provided and appends the meta graph def to the SavedModel proto.

    Args:
      meta_graph_def: The meta graph def to add to the SavedModel.
      tags: The set of tags to annotate the meta graph def with.
      signature_def_map: The map of signature defs to be added to the meta graph
        def.
    """
        for tag in tags:
            meta_graph_def.meta_info_def.tags.append(tag)
        if signature_def_map is not None:
            for key in signature_def_map:
                meta_graph_def.signature_def[key].CopyFrom(signature_def_map[key])
        proto_meta_graph_def = self._saved_model.meta_graphs.add()
        proto_meta_graph_def.CopyFrom(meta_graph_def)

    def _validate_tensor_info(self, tensor_info):
        """Validates the `TensorInfo` proto.

    Checks if the `encoding` (`name` or `coo_sparse` or `type_spec`) and
    `dtype` fields exist and are non-empty.

    Args:
      tensor_info: `TensorInfo` protocol buffer to validate.

    Raises:
      AssertionError: If the `encoding` or `dtype` fields of the supplied
          `TensorInfo` proto are not populated.
    """
        if tensor_info is None:
            raise AssertionError('All TensorInfo protos used in the SignatureDefs must have the name and dtype fields set.')
        if tensor_info.WhichOneof('encoding') is None:
            raise AssertionError(f"Invalid `tensor_info`: {tensor_info}. All TensorInfo protos used in the SignatureDefs must have one of the 'encoding' fields (e.g., name or coo_sparse) set.")
        if tensor_info.WhichOneof('encoding') == 'composite_tensor':
            for component in tensor_info.composite_tensor.components:
                self._validate_tensor_info(component)
        elif tensor_info.dtype == types_pb2.DT_INVALID:
            raise AssertionError(f'Invalid `tensor_info`: {tensor_info}. All TensorInfo protos used in the SignatureDefs must have the dtype field set.')

    def _validate_signature_def_map(self, signature_def_map):
        """Validates the `SignatureDef` entries in the signature def map.

    Validation of entries in the signature def map includes ensuring that the
    `name` and `dtype` fields of the TensorInfo protos of the `inputs` and
    `outputs` of each `SignatureDef` are populated. Also ensures that reserved
    SignatureDef keys for the initialization and train ops are not used.

    Args:
      signature_def_map: The map of signature defs to be validated.

    Raises:
      AssertionError: If a TensorInfo is not valid.
      KeyError: If a reserved signature key is used in the map.
    """
        for signature_def_key in signature_def_map:
            signature_def = signature_def_map[signature_def_key]
            inputs = signature_def.inputs
            outputs = signature_def.outputs
            for inputs_key in inputs:
                self._validate_tensor_info(inputs[inputs_key])
            for outputs_key in outputs:
                self._validate_tensor_info(outputs[outputs_key])
        if constants.INIT_OP_SIGNATURE_KEY in signature_def_map:
            raise KeyError(f'SignatureDef map key "{constants.INIT_OP_SIGNATURE_KEY}" is reserved for initialization. Please use a different key.')
        if constants.TRAIN_OP_SIGNATURE_KEY in signature_def_map:
            raise KeyError(f'SignatureDef map key "{constants.TRAIN_OP_SIGNATURE_KEY}" is reserved for the train op. Please use a different key.')

    def _maybe_create_saver(self, saver=None):
        """Creates a sharded saver if one does not already exist."""
        if not saver:
            saver = tf_saver.Saver(variables._all_saveable_objects(), sharded=True, write_version=saver_pb2.SaverDef.V2, allow_empty=True)
        return saver

    def add_meta_graph(self, tags, signature_def_map=None, assets_list=None, clear_devices=False, init_op=None, train_op=None, saver=None):
        """Adds the current meta graph to the SavedModel.

    Creates a Saver in the current scope and uses the Saver to export the meta
    graph def. Invoking this API requires the `add_meta_graph_and_variables()`
    API to have been invoked before.

    Args:
      tags: The set of tags to annotate the meta graph def with.
      signature_def_map: The map of signature defs to be added to the meta graph
        def.
      assets_list: Assets to be saved with SavedModel. Note
          that this list should be a subset of the assets saved as part of
          the first meta graph in the SavedModel.
      clear_devices: Set to true if the device info on the default graph should
        be cleared.
      init_op: Op or group of ops to execute when the graph is loaded. Note
          that when the init_op is specified it is run after the restore op at
        load-time.
      train_op: Op or group of opts that trains the model when run. This will
        not be run automatically when the graph is loaded, instead saved in
        a SignatureDef accessible through the exported MetaGraph.
      saver: An instance of tf.compat.v1.train.Saver that will be used to export
        the metagraph. If None, a sharded Saver that restores all variables will
        be used.

    Raises:
      AssertionError: If the variables for the SavedModel have not been saved
          yet, or if the graph already contains one or more legacy init ops.
    """
        if not self._has_saved_variables:
            raise AssertionError('Graph state including variables and assets has not been saved yet. Please invoke `add_meta_graph_and_variables()` first.')
        signature_def_map = signature_def_map or {}
        self._validate_signature_def_map(signature_def_map)
        _add_op_to_signature_def_map(signature_def_map, init_op, constants.INIT_OP_SIGNATURE_KEY)
        _add_op_to_signature_def_map(signature_def_map, train_op, constants.TRAIN_OP_SIGNATURE_KEY)
        saver = self._maybe_create_saver(saver)
        meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices, strip_default_attrs=True)
        self._save_and_write_assets(meta_graph_def, assets_list)
        self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

    def add_meta_graph_and_variables(self, sess, tags, signature_def_map=None, assets_list=None, clear_devices=False, init_op=None, train_op=None, strip_default_attrs=False, saver=None):
        """Adds the current meta graph to the SavedModel and saves variables.

    Creates a Saver to save the variables from the provided session. Exports the
    corresponding meta graph def. This function assumes that the variables to be
    saved have been initialized. For a given `SavedModelBuilder`, this API must
    be called exactly once and for the first meta graph to save. For subsequent
    meta graph defs to be added, the `add_meta_graph()` API must be used.

    Args:
      sess: The TensorFlow session from which to save the meta graph and
        variables.
      tags: The set of tags with which to save the meta graph.
      signature_def_map: The map of signature def map to add to the meta graph
        def.
      assets_list: Assets to be saved with SavedModel.
      clear_devices: Set to true if the device info on the default graph should
        be cleared.
      init_op: Op or group of ops to execute when the graph is loaded. Note
          that when the init_op is specified it is run after the restore op at
        load-time.
      train_op: Op or group of ops that trains the model when run. This will
        not be run automatically when the graph is loaded, instead saved in
        a SignatureDef accessible through the exported MetaGraph.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      saver: An instance of tf.compat.v1.train.Saver that will be used to export the
        metagraph and save variables. If None, a sharded Saver that restores
        all variables will be used.

    """
        if self._has_saved_variables:
            raise AssertionError('Graph state including variables and assets has already been saved. Please invoke `add_meta_graph()` instead.')
        signature_def_map = signature_def_map or {}
        self._validate_signature_def_map(signature_def_map)
        _add_op_to_signature_def_map(signature_def_map, init_op, constants.INIT_OP_SIGNATURE_KEY)
        _add_op_to_signature_def_map(signature_def_map, train_op, constants.TRAIN_OP_SIGNATURE_KEY)
        path_helpers.get_or_create_variables_dir(self._export_dir)
        variables_path = path_helpers.get_variables_path(self._export_dir)
        saver = self._maybe_create_saver(saver)
        saver.save(sess, variables_path, write_meta_graph=False, write_state=False)
        meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)
        self._save_and_write_assets(meta_graph_def, assets_list)
        self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)
        self._has_saved_variables = True

    def save(self, as_text=False):
        """Writes a `SavedModel` protocol buffer to disk.

    The function writes the SavedModel protocol buffer to the export directory
    in a serialized format.

    Args:
      as_text: Writes the SavedModel protocol buffer in text format to
        disk. Protocol buffers in text format are useful for debugging, but
        parsing fails when it encounters an unknown field and so is not forward
        compatible. This means changes to TensorFlow may prevent deployment of
        new text format SavedModels to existing serving binaries. Do not deploy
        `as_text` SavedModels to production.

    Returns:
      The path to which the SavedModel protocol buffer was written.
    """
        metrics.IncrementWriteApi(_SAVE_BUILDER_LABEL)
        if not file_io.file_exists(self._export_dir):
            file_io.recursive_create_dir(self._export_dir)
        if as_text:
            path = file_io.join(compat.as_bytes(self._export_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
            file_io.write_string_to_file(path, str(self._saved_model))
        else:
            path = file_io.join(compat.as_bytes(self._export_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
            file_io.write_string_to_file(path, self._saved_model.SerializeToString(deterministic=True))
        tf_logging.info('SavedModel written to: %s', compat.as_text(path))
        metrics.IncrementWrite(write_version='1')
        return path