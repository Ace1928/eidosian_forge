from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from torch.distributed import Store
class RendezvousHandlerRegistry:
    """Represent a registry of :py:class:`RendezvousHandler` backends."""
    _registry: Dict[str, RendezvousHandlerCreator]

    def __init__(self) -> None:
        self._registry = {}

    def register(self, backend: str, creator: RendezvousHandlerCreator) -> None:
        """Register a new rendezvous backend.

        Args:
            backend:
                The name of the backend.
            creator:
                The callback to invoke to construct the
                :py:class:`RendezvousHandler`.
        """
        if not backend:
            raise ValueError('The rendezvous backend name must be a non-empty string.')
        current_creator: Optional[RendezvousHandlerCreator]
        try:
            current_creator = self._registry[backend]
        except KeyError:
            current_creator = None
        if current_creator is not None and current_creator != creator:
            raise ValueError(f"The rendezvous backend '{backend}' cannot be registered with '{creator}' as it is already registered with '{current_creator}'.")
        self._registry[backend] = creator

    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        """Create a new :py:class:`RendezvousHandler`."""
        try:
            creator = self._registry[params.backend]
        except KeyError as e:
            raise ValueError(f"The rendezvous backend '{params.backend}' is not registered. Did you forget to call `{self.register.__name__}`?") from e
        handler = creator(params)
        if handler.get_backend() != params.backend:
            raise RuntimeError(f"The rendezvous backend '{handler.get_backend()}' does not match the requested backend '{params.backend}'.")
        return handler