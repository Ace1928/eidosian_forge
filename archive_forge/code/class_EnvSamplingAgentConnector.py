from typing import Any
from ray.rllib.connectors.connector import (
from ray.rllib.connectors.registry import register_connector
from ray.rllib.utils.typing import AgentConnectorDataType
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class EnvSamplingAgentConnector(AgentConnector):

    def __init__(self, ctx: ConnectorContext, sign=False, limit=None):
        super().__init__(ctx)
        self.observation_space = ctx.observation_space

    def transform(self, ac_data: AgentConnectorDataType) -> AgentConnectorDataType:
        return ac_data

    def to_state(self):
        return (EnvSamplingAgentConnector.__name__, {})

    @staticmethod
    def from_state(ctx: ConnectorContext, params: Any):
        return EnvSamplingAgentConnector(ctx, **params)