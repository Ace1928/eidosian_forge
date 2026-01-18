from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from torch.distributed import Store
class RendezvousHandler(ABC):
    """Main rendezvous interface.

    Note:
        Distributed Torch users normally **do not** need to implement their own
        ``RendezvousHandler``. An implementation based on C10d Store is already
        provided, and is recommended for most users.
    """

    @abstractmethod
    def get_backend(self) -> str:
        """Return the name of the rendezvous backend."""

    @abstractmethod
    def next_rendezvous(self) -> Tuple[Store, int, int]:
        """Main entry-point into the rendezvous barrier.

        Blocks until the rendezvous is complete and the current process is
        included in the formed worker group, or a timeout occurs, or the
        rendezvous was marked closed.

        Returns:
            A tuple of :py:class:`torch.distributed.Store`, ``rank``, and
            ``world size``.

        Raises:
            RendezvousClosedError:
                The rendezvous is closed.
            RendezvousConnectionError:
                The connection to the rendezvous backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
            RendezvousTimeoutError:
                The rendezvous did not complete on time.
        """

    @abstractmethod
    def is_closed(self) -> bool:
        """Check whether the rendezvous has been closed.

        A closed rendezvous means all future attempts to re-rendezvous within
        same job will fail.

        ``is_closed()`` and :py:meth:`set_closed` have semantics of eventual
        propagation and should not be used for synchronization. The intention is
        that if at least one node decides the job is finished, it will close the
        rendezvous, and other nodes will soon observe this and stop running as
        well.
        """

    @abstractmethod
    def set_closed(self):
        """Mark the rendezvous as closed."""

    @abstractmethod
    def num_nodes_waiting(self) -> int:
        """Return the number of nodes who arrived late at the rendezvous
        barrier, hence were not included in the current worker group.

        Callers should periodically call this method to check whether new
        nodes are waiting to join the job and if so admit them by calling
        :py:meth:`next_rendezvous()` (re-rendezvous).
        """

    @abstractmethod
    def get_run_id(self) -> str:
        """Return the run id of the rendezvous.

        The run id is a user-defined id that uniquely identifies an instance of
        a distributed application. It typically maps to a job id and is used to
        allow nodes to join the correct distributed application.
        """

    @abstractmethod
    def shutdown(self) -> bool:
        """Close all resources that were open for the rendezvous.

        Example::

            rdzv_handler = ...
            try:
                store, rank, world_size = rdzv_handler.next_rendezvous()
            finally:
                rdzv_handler.shutdown()
        """