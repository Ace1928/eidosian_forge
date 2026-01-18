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
@estimator_export('estimator.Exporter')
class Exporter(object):
    """A class representing a type of model export."""

    @abc.abstractproperty
    def name(self):
        """Directory name.

    A directory name under the export base directory where exports of
    this type are written.  Should not be `None` nor empty.
    """
        pass

    @abc.abstractmethod
    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        """Exports the given `Estimator` to a specific format.

    Args:
      estimator: the `Estimator` to export.
      export_path: A string containing a directory where to write the export.
      checkpoint_path: The checkpoint path to export.
      eval_result: The output of `Estimator.evaluate` on this checkpoint.
      is_the_final_export: This boolean is True when this is an export in the
        end of training.  It is False for the intermediate exports during the
        training. When passing `Exporter` to `tf.estimator.train_and_evaluate`
        `is_the_final_export` is always False if `TrainSpec.max_steps` is
        `None`.

    Returns:
      The string path to the exported directory or `None` if export is skipped.
    """
        pass