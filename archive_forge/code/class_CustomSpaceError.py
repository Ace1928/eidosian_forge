import warnings
class CustomSpaceError(Exception):
    """The space is a custom gym.Space instance, and is not supported by `AsyncVectorEnv` with `shared_memory=True`."""