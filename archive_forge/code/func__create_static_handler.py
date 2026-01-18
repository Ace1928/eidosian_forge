from .api import RendezvousHandler, RendezvousParameters
from .api import rendezvous_handler_registry as handler_registry
from .dynamic_rendezvous import create_handler
def _create_static_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import static_tcp_rendezvous
    return static_tcp_rendezvous.create_rdzv_handler(params)