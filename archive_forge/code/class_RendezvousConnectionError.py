from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from torch.distributed import Store
class RendezvousConnectionError(RendezvousError):
    """Raised when the connection to a rendezvous backend has failed."""