import os
import sys
import unittest
from typing import Dict, List, Type
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import (
from torch.testing._internal.distributed.pipe_with_ddp_test import (
from torch.testing._internal.distributed.nn.api.remote_module_test import (
from torch.testing._internal.distributed.rpc.dist_autograd_test import (
from torch.testing._internal.distributed.rpc.dist_optimizer_test import (
from torch.testing._internal.distributed.rpc.jit.dist_autograd_test import (
from torch.testing._internal.distributed.rpc.jit.rpc_test import JitRpcTest
from torch.testing._internal.distributed.rpc.jit.rpc_test_faulty import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.distributed.rpc.faulty_agent_rpc_test import (
from torch.testing._internal.distributed.rpc.rpc_test import (
from torch.testing._internal.distributed.rpc.examples.parameter_server_test import ParameterServerTest
from torch.testing._internal.distributed.rpc.examples.reinforcement_learning_rpc_test import (
@unittest.skipIf(TEST_WITH_DEV_DBG_ASAN, 'Skip ASAN as torch + multiprocessing spawn have known issues')
class SpawnHelper(MultiProcessTestCase):

    def setUp(self):
        super().setUp()
        _check_and_set_tcp_init()
        self._spawn_processes()

    def tearDown(self):
        _check_and_unset_tcp_init()
        super().tearDown()