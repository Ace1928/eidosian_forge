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