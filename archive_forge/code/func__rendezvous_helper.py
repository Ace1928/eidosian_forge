import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional
from torch.distributed import FileStore, PrefixStore, Store, TCPStore
from .constants import default_pg_timeout
def _rendezvous_helper(url: str, rank: int, world_size_opt: Optional[int], **kwargs):
    result = urlparse(url)
    if world_size_opt is None:
        world_size = -1
        if result.scheme == 'env':
            rank = int(os.environ.get('RANK', rank))
            world_size = int(os.environ.get('WORLD_SIZE', world_size))
    else:
        world_size = world_size_opt
    if rank != -1 or world_size != -1 or world_size_opt is None:
        query_dict = _query_to_dict(result.query)
        assert 'rank' not in query_dict and 'world_size' not in query_dict, f'The url: {url} has node-specific arguments(rank, world_size) already.'
        if rank != -1:
            query_dict['rank'] = str(rank)
        if world_size != -1 or world_size_opt is None:
            query_dict['world_size'] = str(world_size)
        result = result._replace(query=f'{'&'.join([f'{k}={v}' for k, v in query_dict.items()])}')
        url = urlunparse(result)
    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError(f'No rendezvous handler for {result.scheme}://')
    return _rendezvous_handlers[result.scheme](url, **kwargs)