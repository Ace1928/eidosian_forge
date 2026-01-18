from typing import Any, List
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import tree
from ray.rllib.connectors.agent.synced_filter import SyncedFilterAgentConnector
from ray.rllib.connectors.connector import AgentConnector
from ray.rllib.connectors.connector import (
from ray.rllib.connectors.registry import register_connector
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.filter import MeanStdFilter, ConcurrentMeanStdFilter
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import AgentConnectorDataType
from ray.util.annotations import PublicAPI
from ray.rllib.utils.filter import RunningStat
@PublicAPI(stability='alpha')
class ConcurrentMeanStdObservationFilterAgentConnector(MeanStdObservationFilterAgentConnector):
    """A concurrent version of the MeanStdObservationFilterAgentConnector.

    This version's filter has all operations wrapped by a threading.RLock.
    It can therefore be safely used by multiple threads.
    """

    def __init__(self, ctx: ConnectorContext, demean=True, destd=True, clip=10.0):
        SyncedFilterAgentConnector.__init__(self, ctx)
        filter_shape = tree.map_structure(lambda s: None if isinstance(s, (Discrete, MultiDiscrete)) else np.array(s.shape), get_base_struct_from_space(ctx.observation_space))
        self.filter = ConcurrentMeanStdFilter(filter_shape, demean=True, destd=True, clip=10.0)