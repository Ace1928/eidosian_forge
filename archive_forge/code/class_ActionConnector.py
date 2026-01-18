import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import gymnasium as gym
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class ActionConnector(Connector):
    """Action connector connects policy outputs including actions,
    to user environments.

    An action connector transforms a single piece of policy output in
    ActionConnectorDataType format, which is basically PolicyOutputType plus env and
    agent IDs.

    Any functions that operate directly on PolicyOutputType can be easily adapted
    into an ActionConnector by using register_lambda_action_connector.

    Example:

    .. testcode::

        from ray.rllib.connectors.action.lambdas import (
            register_lambda_action_connector
        )
        ZeroActionConnector = register_lambda_action_connector(
            "ZeroActionsConnector",
            lambda actions, states, fetches: (
                np.zeros_like(actions), states, fetches
            )
        )

    More complicated action connectors can also be implemented by sub-classing
    this ActionConnector class.
    """

    def __call__(self, ac_data: ActionConnectorDataType) -> ActionConnectorDataType:
        """Transform policy output before they are sent to a user environment.

        Args:
            ac_data: Env and agent IDs, plus policy output.

        Returns:
            The processed action connector data.
        """
        return self.transform(ac_data)

    def transform(self, ac_data: ActionConnectorDataType) -> ActionConnectorDataType:
        """Implementation of the actual transform.

        Users should override transform instead of __call__ directly.

        Args:
            ac_data: Env and agent IDs, plus policy output.

        Returns:
            The processed action connector data.
        """
        raise NotImplementedError