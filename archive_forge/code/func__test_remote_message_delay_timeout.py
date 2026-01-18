import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
def _test_remote_message_delay_timeout(self, func, args, dst=None):
    if self.rank != 0:
        return
    dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
    dst_worker = f'worker{dst_rank}'
    rref = rpc.remote(dst_worker, func, args=args, timeout=0.001)
    expected_error = self.get_timeout_error_regex()
    with self.assertRaisesRegex(RuntimeError, expected_error):
        rref._get_future().wait()
    wait_until_pending_futures_and_users_flushed()
    with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
        rref.to_here()
    if dst_rank != self.rank:
        slow_rref = rpc.remote(dst_worker, func, args=args, timeout=2)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            slow_rref.to_here(0.001)
    if dst_rank != self.rank:
        wait_until_owners_and_forks_on_rank(2, 2, rank=dst_rank)