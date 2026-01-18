from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from torch.distributed import Store
class RendezvousError(Exception):
    """Represents the base type for rendezvous errors."""