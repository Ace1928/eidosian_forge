import abc
from dataclasses import dataclass
from typing import List, Any
from torch.futures import Future
from .metadata import (
from .planner import (
class StorageWriter(abc.ABC):
    """
    Interface used by ``save_state_dict`` to write to storage.

    One StorageWriter instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls.

    1) (all ranks) set_up_storage_writer()
    2) (all ranks) prepare_local_plan()
    3) (coordinator) prepare_global_plan()
    4) (all ranks) write_data()
    5) (coordinator) finish()
    """

    @abc.abstractmethod
    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        """
        Initialize this instance.

        Args:
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """
        pass

    @abc.abstractmethod
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recommended
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plan (SavePlan): The local plan from the ``SavePlanner`` in use.

        Returns:
            A transformed ``SavePlan`` after storage local planning
        """
        pass

    @abc.abstractmethod
    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        """
        Perform centralized planning of storage.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the preferred
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plans: A list of ``SavePlan`` instances, one for each rank.

        Returns:
            A list of transformed ``SavePlan`` after storage global planning
        """
        pass

    @abc.abstractmethod
    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[List[WriteResult]]:
        """
        Write all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``SavePlanner::resolve_data`` on each item
        from the plan to get access to the underlying object to write.

        Subclasses should lazily call `resolve_data` as it can allocate memory.
        In case of tensors, make following assumptions:

        - They might be on any device, including not matching the one on ``WriteItem::tensor_data``
        - They might be views or not contiguous. Only the projection needs to be saved.

        Args:
            plan (SavePlan): The save plan to execute.
            planner (SavePlanner): Planner object to be used to resolve items to data.

        Returns:
            A future that completes to a list of WriteResult
        """
        pass

    @abc.abstractmethod
    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        """
        Write the metadata and marks the current checkpoint as successful.

        The actual format/schema used for serializing `metadata` is an
        implementation detail. The only requirement is that it's recoverable
        in to the same object graph.

        Args:
            metadata (Metadata): metadata for the new checkpoint
            results: A list of WriteResults from all ranks.

        Returns:
            None
        """
        pass