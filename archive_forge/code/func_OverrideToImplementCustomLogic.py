from ray.rllib.utils.deprecation import Deprecated
from ray.util.annotations import _mark_annotated
def OverrideToImplementCustomLogic(obj):
    """Users should override this in their sub-classes to implement custom logic.

    Used in Algorithm and Policy to tag methods that need overriding, e.g.
    `Policy.loss()`.

    .. testcode::
        :skipif: True

        from ray.rllib.policy.torch_policy import TorchPolicy
        @overrides(TorchPolicy)
        @OverrideToImplementCustomLogic
        def loss(self, ...):
            # implement custom loss function here ...
            # ... w/o calling the corresponding `super().loss()` method.
            ...

    """
    obj.__is_overriden__ = False
    return obj