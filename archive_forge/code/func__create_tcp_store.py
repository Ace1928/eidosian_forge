import binascii
import logging
import os
import tempfile
from base64 import b64decode, b64encode
from datetime import timedelta
from typing import Any, Optional, Tuple, cast
from torch.distributed import FileStore, Store, TCPStore
from torch.distributed.elastic.events import (
from .api import (
from .dynamic_rendezvous import RendezvousBackend, Token
from .utils import _matches_machine_hostname, parse_rendezvous_endpoint
def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=29400)
    cfg_is_host = params.get_as_bool('is_host')
    if cfg_is_host is not None:
        is_host = cfg_is_host
    else:
        is_host = _matches_machine_hostname(host)
    read_timeout = cast(int, params.get_as_int('read_timeout', 60))
    if read_timeout <= 0:
        raise ValueError('The read timeout must be a positive integer.')
    for is_server in [is_host, False]:
        try:
            store = TCPStore(host, port, is_master=is_server, timeout=timedelta(seconds=read_timeout))
            if is_server:
                msg = f'Process {os.getpid()} hosts the TCP store for the C10d rendezvous backend.'
                construct_and_record_rdzv_event(run_id=params.run_id, message=msg, node_state=NodeState.INIT)
                log.info(msg)
            break
        except (ValueError, RuntimeError, TimeoutError) as exc:
            if not is_server or cfg_is_host is not None:
                raise RendezvousConnectionError('The connection to the C10d store has failed. See inner exception for details.') from exc
    return store