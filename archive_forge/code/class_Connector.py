import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import gymnasium as gym
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class Connector(abc.ABC):
    """Connector base class.

    A connector is a step of transformation, of either envrionment data before they
    get to a policy, or policy output before it is sent back to the environment.

    Connectors may be training-aware, for example, behave slightly differently
    during training and inference.

    All connectors are required to be serializable and implement to_state().
    """

    def __init__(self, ctx: ConnectorContext):
        self._is_training = True

    def in_training(self):
        self._is_training = True

    def in_eval(self):
        self._is_training = False

    def __str__(self, indentation: int=0):
        return ' ' * indentation + self.__class__.__name__

    def to_state(self) -> Tuple[str, Any]:
        """Serialize a connector into a JSON serializable Tuple.

        to_state is required, so that all Connectors are serializable.

        Returns:
            A tuple of connector's name and its serialized states.
            String should match the name used to register the connector,
            while state can be any single data structure that contains the
            serialized state of the connector. If a connector is stateless,
            state can simply be None.
        """
        return NotImplementedError

    @staticmethod
    def from_state(self, ctx: ConnectorContext, params: Any) -> 'Connector':
        """De-serialize a JSON params back into a Connector.

        from_state is required, so that all Connectors are serializable.

        Args:
            ctx: Context for constructing this connector.
            params: Serialized states of the connector to be recovered.

        Returns:
            De-serialized connector.
        """
        return NotImplementedError