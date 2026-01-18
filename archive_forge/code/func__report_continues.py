import numpy as np
from ray.rllib.algorithms.dreamerv3.utils.debugging import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_utils import inverse_symlog
def _report_continues(*, results, computed_continues, sampled_continues, descr_prefix=None, descr_cont):
    descr_prefix = descr_prefix + '_' if descr_prefix else ''
    mse_sampled_vs_computed_continues = np.mean(np.square(computed_continues - sampled_continues.astype(computed_continues.dtype)))
    results.update({f'{descr_prefix}sampled_vs_{descr_cont}_continues_mse': mse_sampled_vs_computed_continues})