import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic import events, metrics
from torch.distributed.elastic.agent.server.api import WorkerSpec
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing import SignalException, Std
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint
from torch.distributed.elastic.utils.logging import get_logger
def _get_addr_and_port(rdzv_parameters: RendezvousParameters) -> Tuple[Optional[str], Optional[int]]:
    if rdzv_parameters.backend != 'static':
        return (None, None)
    endpoint = rdzv_parameters.endpoint
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError('Endpoint is missing in endpoint. Try to add --master-addr and --master-port')
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, default_port=-1)
    if master_port == -1:
        raise ValueError(f'port is missing in endpoint: {endpoint}. Try to specify --master-port')
    return (master_addr, master_port)