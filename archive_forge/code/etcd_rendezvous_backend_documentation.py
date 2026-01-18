import binascii
from base64 import b64decode, b64encode
from typing import Optional, Tuple, cast
import urllib3.exceptions  # type: ignore[import]
from etcd import Client as EtcdClient  # type: ignore[import]
from etcd import (
from torch.distributed import Store
from .api import RendezvousConnectionError, RendezvousParameters, RendezvousStateError
from .dynamic_rendezvous import RendezvousBackend, Token
from .etcd_store import EtcdStore
from .utils import parse_rendezvous_endpoint
See base class.