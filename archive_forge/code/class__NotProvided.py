from copy import deepcopy
from functools import partial
import importlib
import json
import os
import re
import yaml
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils import force_list, merge_dicts
class _NotProvided:
    """Singleton class to provide a "not provided" value for AlgorithmConfig signatures.

    Using the only instance of this class indicates that the user does NOT wish to
    change the value of some property.

    .. testcode::
        :skipif: True

        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
        config = AlgorithmConfig()
        # Print out the default learning rate.
        print(config.lr)

    .. testoutput::

        0.001

    .. testcode::
        :skipif: True

        # Print out the default `preprocessor_pref`.
        print(config.preprocessor_pref)

    .. testoutput::

        "deepmind"

    .. testcode::
        :skipif: True

        # Will only set the `preprocessor_pref` property (to None) and leave
        # all other properties at their default values.
        config.training(preprocessor_pref=None)
        config.preprocessor_pref is None

    .. testoutput::

        True

    .. testcode::
        :skipif: True

        # Still the same value (didn't touch it in the call to `.training()`.
        print(config.lr)

    .. testoutput::

        0.001
    """

    class __NotProvided:
        pass
    instance = None

    def __init__(self):
        if _NotProvided.instance is None:
            _NotProvided.instance = _NotProvided.__NotProvided()