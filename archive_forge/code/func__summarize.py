import numpy as np
from ray.rllib.algorithms.dreamerv3.utils.debugging import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_utils import inverse_symlog
def _summarize(*, results, data_to_summarize, keys_to_log, include_histograms=False):
    for k in keys_to_log:
        if data_to_summarize[k].shape == ():
            results.update({k: data_to_summarize[k]})
        elif include_histograms:
            results.update({k: data_to_summarize[k]})