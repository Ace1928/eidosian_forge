import numpy as np
from ray.rllib.algorithms.dreamerv3.utils.debugging import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_utils import inverse_symlog
def _report_obs(*, results, computed_float_obs_B_T_dims, sampled_obs_B_T_dims, descr_prefix=None, descr_obs, symlog_obs):
    """Summarizes computed- vs sampled observations: MSE and (if applicable) images.

    Args:
        computed_float_obs_B_T_dims: Computed float observations
            (not clipped, not cast'd). Shape=(B, T, [dims ...]).
        sampled_obs_B_T_dims: Sampled observations (as-is from the environment, meaning
            this could be uint8, 0-255 clipped images). Shape=(B, T, [dims ...]).
        B: The batch size B (see shapes of `computed_float_obs_B_T_dims` and
            `sampled_obs_B_T_dims` above).
        T: The batch length T (see shapes of `computed_float_obs_B_T_dims` and
            `sampled_obs_B_T_dims` above).
        descr: A string used to describe the computed data to be used in the TB
            summaries.
    """
    if len(sampled_obs_B_T_dims.shape) in [4, 5]:
        descr_prefix = descr_prefix + '_' if descr_prefix else ''
        if symlog_obs:
            computed_float_obs_B_T_dims = inverse_symlog(computed_float_obs_B_T_dims)
        if not symlog_obs:
            computed_float_obs_B_T_dims = (computed_float_obs_B_T_dims + 1.0) * 128
            sampled_obs_B_T_dims = (sampled_obs_B_T_dims + 1.0) * 128
            sampled_obs_B_T_dims = np.clip(sampled_obs_B_T_dims, 0.0, 255.0).astype(np.uint8)
        computed_images = np.clip(computed_float_obs_B_T_dims, 0.0, 255.0).astype(np.uint8)
        sampled_vs_computed_images = np.concatenate([computed_images, sampled_obs_B_T_dims], axis=3)
        if len(sampled_obs_B_T_dims.shape) == 2 + 2:
            sampled_vs_computed_images = np.expand_dims(sampled_vs_computed_images, -1)
        results.update({f'{descr_prefix}sampled_vs_{descr_obs}_videos': sampled_vs_computed_images})