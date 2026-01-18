import os
import sys
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class SavedModelLoader(object):
    """Load graphs and restore variable values from a `SavedModel`."""

    def __init__(self, export_dir):
        """Creates a `SavedModelLoader`.

    Args:
      export_dir: Directory in which the SavedModel protocol buffer and
        variables to be loaded are located.
    """
        self._export_dir = export_dir
        self._variables_path = path_helpers.get_variables_path(export_dir)
        self._saved_model = parse_saved_model(export_dir)

    @property
    def export_dir(self):
        """Directory containing the SavedModel."""
        return self._export_dir

    @property
    def variables_path(self):
        """Path to variable checkpoint files."""
        return self._variables_path

    @property
    def saved_model(self):
        """SavedModel object parsed from the export directory."""
        return self._saved_model

    def get_meta_graph_def_from_tags(self, tags):
        """Return MetaGraphDef with the exact specified tags.

    Args:
      tags: A list or set of string tags that identify the MetaGraphDef.

    Returns:
      MetaGraphDef with the same tags.

    Raises:
      RuntimeError: if no metagraphs were found with the associated tags.
    """
        found_match = False
        meta_graph_def_to_load = None
        available_tags = []
        for meta_graph_def in self._saved_model.meta_graphs:
            available_tags.append(set(meta_graph_def.meta_info_def.tags))
            if set(meta_graph_def.meta_info_def.tags) == set(tags):
                meta_graph_def_to_load = meta_graph_def
                found_match = True
                break
        if not found_match:
            raise RuntimeError(f"MetaGraphDef associated with tags {str(tags).strip('[]')} could not be found in SavedModel, with available tags '{available_tags}'. To inspect available tag-sets in the SavedModel, please use the SavedModel CLI: `saved_model_cli`.")
        return meta_graph_def_to_load

    def load_graph(self, graph, tags, import_scope=None, **saver_kwargs):
        """Load ops and nodes from SavedModel MetaGraph into graph.

    Args:
      graph: tf.Graph object.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
      **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.

    Returns:
      A tuple of
        * Saver defined by the MetaGraph, which can be used to restore the
          variable values.
        * List of `Operation`/`Tensor` objects returned from
          `tf.import_graph_def` (may be `None`).
    """
        meta_graph_def = self.get_meta_graph_def_from_tags(tags)
        if sys.byteorder == 'big':
            saved_model_utils.swap_function_tensor_content(meta_graph_def, 'little', 'big')
        with graph.as_default():
            return tf_saver._import_meta_graph_with_return_elements(meta_graph_def, import_scope=import_scope, **saver_kwargs)

    def restore_variables(self, sess, saver, import_scope=None):
        """Restore SavedModel variable values into the session.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      saver: a tf.compat.v1.train.Saver object. Can be None if there are no
        variables in graph. This may be the saver returned by the load_graph()
        function, or a default `tf.compat.v1.train.Saver()`.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.

    Raises:
      ValueError: if no saver was passed to the saver argument, and there are
        variables in the graph.
    """
        with sess.graph.as_default():
            if saver is None and (not variables._all_saveable_objects(scope=import_scope)):
                tf_logging.info('The specified SavedModel has no variables; no checkpoints were restored.')
            elif isinstance(saver, tf_saver.Saver):
                saver.restore(sess, self._variables_path)
            else:
                raise ValueError('No tf.train.Saver object was passed to the function `SavedModelLoader.restore_variables`. Since there are variables in the graph, a saver is required.')

    def run_init_ops(self, sess, tags, import_scope=None):
        """Run initialization ops defined in the `MetaGraphDef`.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
    """
        meta_graph_def = self.get_meta_graph_def_from_tags(tags)
        with sess.graph.as_default():
            asset_tensors_dictionary = get_asset_tensors(self._export_dir, meta_graph_def, import_scope=import_scope)
            init_op = get_init_op(meta_graph_def, import_scope)
            if init_op is not None:
                sess.run(fetches=[init_op], feed_dict=asset_tensors_dictionary)

    def load(self, sess, tags, import_scope=None, **saver_kwargs):
        """Load the MetaGraphDef graph and restore variable values into the session.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
      **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.

    Returns:
      `MetagraphDef` proto of the graph that was loaded.
    """
        saved_model_proto = parse_saved_model(self._export_dir)
        metrics.IncrementReadApi(_LOADER_LABEL)
        with sess.graph.as_default():
            saver, _ = self.load_graph(sess.graph, tags, import_scope, **saver_kwargs)
            self.restore_variables(sess, saver, import_scope)
            self.run_init_ops(sess, tags, import_scope)
        meta_graph_def = self.get_meta_graph_def_from_tags(tags)
        if len(saved_model_proto.meta_graphs) == 1 and saved_model_proto.meta_graphs[0].HasField('object_graph_def'):
            metrics.IncrementRead(write_version='2')
        else:
            metrics.IncrementRead(write_version='1')
        return meta_graph_def