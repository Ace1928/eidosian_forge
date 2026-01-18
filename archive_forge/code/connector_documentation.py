import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import gymnasium as gym
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
Returns a list of connectors that fit 'key'.

        If key is a number n, we return a list with the nth element of this pipeline.
        If key is a Connector class or a string matching the class name of a
        Connector class, we return a list of all connectors in this pipeline matching
        the specified class.

        Args:
            key: The key to index by

        Returns: The Connector at index `key`.
        