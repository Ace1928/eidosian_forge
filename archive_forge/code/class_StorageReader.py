import abc
from dataclasses import dataclass
from typing import List, Any
from torch.futures import Future
from .metadata import (
from .planner import (
class StorageReader(abc.ABC):
    """
    Interface used by ``load_state_dict`` to read from storage.

    One StorageReader instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expected the following sequence of calls by ``load_state_dict``:

    1) (all ranks) read_metadata()
    2) (all ranks) set_up_storage_reader()
    3) (all ranks) prepare_local_plan()
    4) (coordinator) prepare_global_plan()
    5) (all ranks) read_data()
    """

    @abc.abstractmethod
    def read_metadata(self) -> Metadata:
        """
        Read the checkpoint metadata.

        Returns:
            The metadata object associated with the checkpoint being loaded.

        """
        pass

    @abc.abstractmethod
    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        """
        Initialize this instance.

        Args:
            metadata (Metadata): The metadata schema to use.
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """
        pass

    @abc.abstractmethod
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recommended
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plan (LoadPlan): The local plan from the ``LoadPlan`` in use.

        Returns:
            A transformed ``LoadPlan`` after storage local planning
        """
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        """
        Perform centralized planning of storage loading.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the preferred
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plans: A list of ``LoadPlan`` instances, one for each rank.

        Returns:
            A list of transformed ``LoadPlan`` after storage global planning
        """
        pass

    @abc.abstractmethod
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Read all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``LoadPlanner::load_bytes`` to deserialize a BytesIO
        object into the right place.

        A subclass should call ``LoadPlanner::resolve_tensor`` to get access to the
        tensors that in should load data into.

        It's the StorageLayer responsibility to properly schedule any cross device copies
        required.

        Args:
            plan (LoadPlan): The local plan to execute on
            planner (LoadPlanner): The planner object to use to resolve items.

        Returns:
            A future that completes once all reads are finished.
        """
        pass