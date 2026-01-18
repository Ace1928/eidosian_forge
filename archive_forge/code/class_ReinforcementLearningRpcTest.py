import numpy as np
from itertools import count
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
class ReinforcementLearningRpcTest(RpcAgentTestFixture):

    @dist_init(setup_rpc=False)
    def test_rl_rpc(self):
        if self.rank == 0:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
            agent = Agent(self.world_size)
            run_agent(agent, n_steps=int(TOTAL_EPISODE_STEP / (self.world_size - 1)))
            self.assertGreater(agent.running_reward, 0.0)
        else:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        rpc.shutdown()