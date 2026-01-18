from dataclasses import dataclass
from typing import Any, DefaultDict, Dict
from ray.rllib.core.learner.learner import Learner, LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
@dataclass
class DreamerV3LearnerHyperparameters(LearnerHyperparameters):
    """Hyperparameters for the DreamerV3Learner sub-classes (framework specific).

    These should never be set directly by the user. Instead, use the DreamerV3Config
    class to configure your algorithm.
    See `ray.rllib.algorithms.dreamerv3.dreamerv3::DreamerV3Config::training()` for
    more details on the individual properties.
    """
    model_size: str = None
    training_ratio: float = None
    batch_size_B: int = None
    batch_length_T: int = None
    horizon_H: int = None
    gamma: float = None
    gae_lambda: float = None
    entropy_scale: float = None
    return_normalization_decay: float = None
    world_model_lr: float = None
    actor_lr: float = None
    critic_lr: float = None
    train_critic: bool = None
    train_actor: bool = None
    use_curiosity: bool = None
    intrinsic_rewards_scale: float = None
    world_model_grad_clip_by_global_norm: float = None
    actor_grad_clip_by_global_norm: float = None
    critic_grad_clip_by_global_norm: float = None
    use_float16: bool = None
    report_individual_batch_item_stats: bool = None
    report_dream_data: bool = None
    report_images_and_videos: bool = None