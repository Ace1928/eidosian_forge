import datetime
import random
import time
from base64 import b64decode, b64encode
from typing import Optional
import etcd  # type: ignore[import]
from torch.distributed import Store
def _try_wait_get(self, b64_keys, override_timeout=None):
    timeout = self.timeout if override_timeout is None else override_timeout
    deadline = time.time() + timeout.total_seconds()
    while True:
        all_nodes = self.client.get(key=self.prefix)
        req_nodes = {node.key: node.value for node in all_nodes.children if node.key in b64_keys}
        if len(req_nodes) == len(b64_keys):
            return req_nodes
        watch_timeout = deadline - time.time()
        if watch_timeout <= 0:
            return None
        try:
            self.client.watch(key=self.prefix, recursive=True, timeout=watch_timeout, index=all_nodes.etcd_index + 1)
        except etcd.EtcdWatchTimedOut:
            if time.time() >= deadline:
                return None
            else:
                continue
        except etcd.EtcdEventIndexCleared:
            continue