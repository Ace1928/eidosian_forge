from typing import Any, Callable, Dict, Type
from ray.rllib.connectors.connector import (
from ray.rllib.connectors.registry import register_connector
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
class LambdaActionConnector(ActionConnector):

    def transform(self, ac_data: ActionConnectorDataType) -> ActionConnectorDataType:
        assert isinstance(ac_data.output, tuple), 'Action connector requires PolicyOutputType data.'
        actions, states, fetches = ac_data.output
        return ActionConnectorDataType(ac_data.env_id, ac_data.agent_id, ac_data.input_dict, fn(actions, states, fetches))

    def to_state(self):
        return (name, None)

    @staticmethod
    def from_state(ctx: ConnectorContext, params: Any):
        return LambdaActionConnector(ctx)