from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import gc
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@estimator_export('estimator.LatestExporter')
class LatestExporter(Exporter):
    """This class regularly exports the serving graph and checkpoints.

  In addition to exporting, this class also garbage collects stale exports.
  """

    def __init__(self, name, serving_input_receiver_fn, assets_extra=None, as_text=False, exports_to_keep=5):
        """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
        collection.

    Raises:
      ValueError: if any arguments is invalid.
    """
        self._saved_model_exporter = _SavedModelExporter(name, serving_input_receiver_fn, assets_extra, as_text)
        self._exports_to_keep = exports_to_keep
        if exports_to_keep is not None and exports_to_keep <= 0:
            raise ValueError('`exports_to_keep`, if provided, must be positive number')

    @property
    def name(self):
        return self._saved_model_exporter.name

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        export_result = self._saved_model_exporter.export(estimator, export_path, checkpoint_path, eval_result, is_the_final_export)
        self._garbage_collect_exports(export_path)
        return export_result

    def _garbage_collect_exports(self, export_dir_base):
        """Deletes older exports, retaining only a given number of the most recent.

    Export subdirectories are assumed to be named with monotonically increasing
    integers; the most recent are taken to be those with the largest values.

    Args:
      export_dir_base: the base directory under which each export is in a
        versioned subdirectory.
    """
        if self._exports_to_keep is None:
            return

        def _export_version_parser(path):
            filename = os.path.basename(path.path)
            if not (len(filename) == 10 and filename.isdigit()):
                return None
            return path._replace(export_version=int(filename))
        keep_filter = gc._largest_export_versions(self._exports_to_keep)
        delete_filter = gc._negation(keep_filter)
        for p in delete_filter(gc._get_paths(export_dir_base, parser=_export_version_parser)):
            try:
                tf.compat.v1.gfile.DeleteRecursively(p.path)
            except tf.errors.NotFoundError as e:
                tf.compat.v1.logging.warn('Can not delete %s recursively: %s', p.path, e)