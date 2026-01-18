from ray.rllib.utils.deprecation import Deprecated
from ray.util.annotations import _mark_annotated
def DeveloperAPI(obj):
    """Decorator for documenting developer APIs.

    Developer APIs are classes and methods explicitly exposed to developers
    for the purposes of building custom algorithms or advanced training
    strategies on top of RLlib internals. You can generally expect these APIs
    to be stable sans minor changes (but less stable than public APIs).

    Subclasses that inherit from a ``@DeveloperAPI`` base class can be
    assumed part of the RLlib developer API as well.

    .. testcode::
        :skipif: True

        # Indicates that the `TorchPolicy` class is exposed to end users
        # of RLlib and will remain (relatively) stable across RLlib
        # releases.
        from ray.rllib.policy import Policy
        @DeveloperAPI
        class TorchPolicy(Policy):
            ...
    """
    _mark_annotated(obj)
    return obj