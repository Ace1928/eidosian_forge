from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from torch.distributed import Store
class RendezvousStateError(RendezvousError):
    """Raised when the state of a rendezvous is corrupt."""