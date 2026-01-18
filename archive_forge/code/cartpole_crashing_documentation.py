import logging
from gymnasium.envs.classic_control import CartPoleEnv
import numpy as np
import time
from ray.rllib.examples.env.multi_agent import make_multi_agent
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import EnvError
A CartPole env that crashes from time to time.

    Useful for testing faulty sub-env (within a vectorized env) handling by
    RolloutWorkers.

    After crashing, the env expects a `reset()` call next (calling `step()` will
    result in yet another error), which may or may not take a very long time to
    complete. This simulates the env having to reinitialize some sub-processes, e.g.
    an external connection.
    