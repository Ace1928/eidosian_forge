import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.data import provider
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.pr_curve import metadata
def _make_pr_entry(self, step, wall_time, data_array):
    """Creates an entry for PR curve data. Each entry corresponds to 1
        step.

        Args:
          step: The step.
          wall_time: The wall time.
          data_array: A numpy array of PR curve data stored in the summary format.

        Returns:
          A PR curve entry.
        """
    tp_index = metadata.TRUE_POSITIVES_INDEX
    fp_index = metadata.FALSE_POSITIVES_INDEX
    tn_index = metadata.TRUE_NEGATIVES_INDEX
    fn_index = metadata.FALSE_NEGATIVES_INDEX
    positives = data_array[[tp_index, fp_index], :].astype(int).sum(axis=0)
    end_index_inclusive = len(positives) - 1
    while end_index_inclusive > 0 and positives[end_index_inclusive] == 0:
        end_index_inclusive -= 1
    end_index = end_index_inclusive + 1
    num_thresholds = data_array.shape[1]
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    true_positives = [int(v) for v in data_array[tp_index]]
    false_positives = [int(v) for v in data_array[fp_index]]
    true_negatives = [int(v) for v in data_array[tn_index]]
    false_negatives = [int(v) for v in data_array[fn_index]]
    return {'wall_time': wall_time, 'step': step, 'precision': data_array[metadata.PRECISION_INDEX, :end_index].tolist(), 'recall': data_array[metadata.RECALL_INDEX, :end_index].tolist(), 'true_positives': true_positives[:end_index], 'false_positives': false_positives[:end_index], 'true_negatives': true_negatives[:end_index], 'false_negatives': false_negatives[:end_index], 'thresholds': thresholds[:end_index].tolist()}