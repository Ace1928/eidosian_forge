import gymnasium as gym
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.base import Encoder, ActorCriticEncoder, Model
from ray.rllib.utils import override
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
def _check_if_diag_gaussian(action_distribution_cls, framework):
    if framework == 'torch':
        from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
        assert issubclass(action_distribution_cls, TorchDiagGaussian), f'free_log_std is only supported for DiagGaussian action distributions. Found action distribution: {action_distribution_cls}.'
    elif framework == 'tf2':
        from ray.rllib.models.tf.tf_distributions import TfDiagGaussian
        assert issubclass(action_distribution_cls, TfDiagGaussian), 'free_log_std is only supported for DiagGaussian action distributions. Found action distribution: {}.'.format(action_distribution_cls)
    else:
        raise ValueError(f'Framework {framework} not supported for free_log_std.')