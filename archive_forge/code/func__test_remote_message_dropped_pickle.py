import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
def _test_remote_message_dropped_pickle(self, dst=None):
    if self.rank != 0:
        return
    dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
    dst_worker = f'worker{dst_rank}'
    rref = rpc.remote(dst_worker, my_sleep_func, args=(1,))
    wait_until_pending_futures_and_users_flushed()
    with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
        rref._serialize()
    with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
        rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 1))