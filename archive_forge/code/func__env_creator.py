from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray import tune
def _env_creator(ctx):
    import flappy_bird_gymnasium
    import gymnasium as gym
    from supersuit.generic_wrappers import resize_v1
    from ray.rllib.algorithms.dreamerv3.utils.env_runner import NormalizedImageEnv
    return NormalizedImageEnv(resize_v1(gym.make('FlappyBird-rgb-v0', audio_on=False), x_size=64, y_size=64))