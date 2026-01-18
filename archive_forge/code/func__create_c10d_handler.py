from .api import RendezvousHandler, RendezvousParameters
from .api import rendezvous_handler_registry as handler_registry
from .dynamic_rendezvous import create_handler
def _create_c10d_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .c10d_rendezvous_backend import create_backend
    backend, store = create_backend(params)
    return create_handler(store, backend, params)