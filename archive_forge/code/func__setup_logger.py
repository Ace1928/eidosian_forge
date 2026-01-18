import logging
from ray._private.usage import usage_lib
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.tune.registry import register_trainable
def _setup_logger():
    logger = logging.getLogger('ray.rllib')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s'))
    logger.addHandler(handler)
    logger.propagate = False