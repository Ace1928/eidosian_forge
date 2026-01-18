import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional
from torch.distributed import FileStore, PrefixStore, Store, TCPStore
from .constants import default_pg_timeout
def _env_rendezvous_handler(url: str, timeout: timedelta=default_pg_timeout, **kwargs):

    def _error(msg):
        return _rendezvous_error('env:// rendezvous: ' + msg)

    def _env_error(var):
        return _error(f'environment variable {var} expected, but not set')

    def _get_env_or_raise(env_var: str) -> str:
        env_val = os.environ.get(env_var, None)
        if not env_val:
            raise _env_error(env_var)
        else:
            return env_val
    result = urlparse(url)
    query_dict = _query_to_dict(result.query)
    rank: int
    world_size: int
    master_port: int
    master_addr: str
    if 'rank' in query_dict:
        rank = int(query_dict['rank'])
    else:
        rank = int(_get_env_or_raise('RANK'))
    if 'world_size' in query_dict:
        world_size = int(query_dict['world_size'])
    else:
        world_size = int(_get_env_or_raise('WORLD_SIZE'))
    master_addr = _get_env_or_raise('MASTER_ADDR')
    master_port = int(_get_env_or_raise('MASTER_PORT'))
    use_libuv = query_dict.get('use_libuv', os.environ.get('USE_LIBUV', '0')) == '1'
    store = _create_c10d_store(master_addr, master_port, rank, world_size, timeout, use_libuv)
    yield (store, rank, world_size)
    raise RuntimeError('Unable to perform re-rendezvous using env:// method')