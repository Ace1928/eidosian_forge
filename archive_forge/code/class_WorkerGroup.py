import logging
import os
import socket
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import exception_cause, skip_exceptions
from ray.types import ObjectRef
from ray.util.placement_group import PlacementGroup
class WorkerGroup:
    """Group of Ray Actors that can execute arbitrary functions.

    ``WorkerGroup`` launches Ray actors according to the given
    specification. It can then execute arbitrary Python functions in each of
    these workers.

    If not enough resources are available to launch the actors, the Ray
    cluster will automatically scale up if autoscaling is enabled.

    Args:
        num_workers: The number of workers (Ray actors) to launch.
            Defaults to 1.
        num_cpus_per_worker: The number of CPUs to reserve for each
            worker. Fractional values are allowed. Defaults to 1.
        num_gpus_per_worker: The number of GPUs to reserve for each
            worker. Fractional values are allowed. Defaults to 0.
        additional_resources_per_worker (Optional[Dict[str, float]]):
            Dictionary specifying the extra resources that will be
            requested for each worker in addition to ``num_cpus_per_worker``
            and ``num_gpus_per_worker``.
        actor_cls (Optional[Type]): If specified use this class as the
            remote actors.
        remote_cls_args, remote_cls_kwargs: If ``remote_cls`` is provided,
            these args will be used for the worker initialization.
        placement_group (PlacementGroup|str): The placement group that workers
            should be created in. Defaults to "default" which will inherit the
            parent placement group (if child tasks should be captured).


    Example:

    .. code_block:: python

        worker_group = WorkerGroup(num_workers=2)
        output = worker_group.execute(lambda: 1)
        assert len(output) == 2
        assert all(o == 1 for o in output)
    """

    def __init__(self, num_workers: int=1, num_cpus_per_worker: float=1, num_gpus_per_worker: float=0, additional_resources_per_worker: Optional[Dict[str, float]]=None, actor_cls: Type=None, actor_cls_args: Optional[Tuple]=None, actor_cls_kwargs: Optional[Dict]=None, placement_group: Union[PlacementGroup, str]='default'):
        if num_workers <= 0:
            raise ValueError(f'The provided `num_workers` must be greater than 0. Received num_workers={num_workers} instead.')
        if num_cpus_per_worker < 0 or num_gpus_per_worker < 0:
            raise ValueError(f'The number of CPUs and GPUs per worker must not be negative. Received num_cpus_per_worker={num_cpus_per_worker} and num_gpus_per_worker={num_gpus_per_worker}.')
        if (actor_cls_args or actor_cls_kwargs) and (not actor_cls):
            raise ValueError('`actor_cls_args` or `actor_class_kwargs` are passed in but no `actor_cls` is passed in.')
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = num_gpus_per_worker
        self.additional_resources_per_worker = additional_resources_per_worker
        self.workers = []
        self._base_cls = create_executable_class(actor_cls)
        assert issubclass(self._base_cls, RayTrainWorker)
        self._actor_cls_args = actor_cls_args or []
        self._actor_cls_kwargs = actor_cls_kwargs or {}
        self._placement_group = placement_group
        self._remote_cls = ray.remote(num_cpus=self.num_cpus_per_worker, num_gpus=self.num_gpus_per_worker, resources=self.additional_resources_per_worker)(self._base_cls)
        self.start()

    def start(self):
        """Starts all the workers in this worker group."""
        if self.workers and len(self.workers) > 0:
            raise RuntimeError('The workers have already been started. Please call `shutdown` first if you want to restart them.')
        logger.debug(f'Starting {self.num_workers} workers.')
        self.add_workers(self.num_workers)
        logger.debug(f'{len(self.workers)} workers have successfully started.')

    def shutdown(self, patience_s: float=5):
        """Shutdown all the workers in this worker group.

        Args:
            patience_s: Attempt a graceful shutdown
                of the workers for this many seconds. Fallback to force kill
                if graceful shutdown is not complete after this time. If
                this is less than or equal to 0, immediately force kill all
                workers.
        """
        logger.debug(f'Shutting down {len(self.workers)} workers.')
        if patience_s <= 0:
            for worker in self.workers:
                ray.kill(worker.actor)
        else:
            done_refs = [w.actor.__ray_terminate__.remote() for w in self.workers]
            done, not_done = ray.wait(done_refs, timeout=patience_s)
            if not_done:
                logger.debug('Graceful termination failed. Falling back to force kill.')
                for worker in self.workers:
                    ray.kill(worker.actor)
        logger.debug('Shutdown successful.')
        self.workers = []

    def execute_async(self, func: Callable[..., T], *args, **kwargs) -> List[ObjectRef]:
        """Execute ``func`` on each worker and return the futures.

        Args:
            func: A function to call on each worker.
            args, kwargs: Passed directly into func.

        Returns:
            (List[ObjectRef]) A list of ``ObjectRef`` representing the
                output of ``func`` from each worker. The order is the same
                as ``self.workers``.

        """
        if len(self.workers) <= 0:
            raise RuntimeError('There are no active workers. This worker group has most likely been shut down. Pleasecreate a new WorkerGroup or restart this one.')
        return [w.actor._RayTrainWorker__execute.options(name=f'_RayTrainWorker__execute.{func.__name__}').remote(func, *args, **kwargs) for w in self.workers]

    def execute(self, func: Callable[..., T], *args, **kwargs) -> List[T]:
        """Execute ``func`` on each worker and return the outputs of ``func``.

        Args:
            func: A function to call on each worker.
            args, kwargs: Passed directly into func.

        Returns:
            (List[T]) A list containing the output of ``func`` from each
                worker. The order is the same as ``self.workers``.

        """
        return ray.get(self.execute_async(func, *args, **kwargs))

    def execute_single_async(self, worker_index: int, func: Callable[..., T], *args, **kwargs) -> ObjectRef:
        """Execute ``func`` on worker ``worker_index`` and return futures.

        Args:
            worker_index: The index to execute func on.
            func: A function to call on the first worker.
            args, kwargs: Passed directly into func.

        Returns:
            (ObjectRef) An ObjectRef representing the output of func.

        """
        if worker_index >= len(self.workers):
            raise ValueError(f'The provided worker_index {worker_index} is not valid for {self.num_workers} workers.')
        return self.workers[worker_index].actor._RayTrainWorker__execute.options(name=f'_RayTrainWorker__execute.{func.__name__}').remote(func, *args, **kwargs)

    def execute_single(self, worker_index: int, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute ``func`` on worker with index ``worker_index``.

        Args:
            worker_index: The index to execute func on.
            func: A function to call on the first worker.
            args, kwargs: Passed directly into func.

        Returns:
            (T) The output of func.

        """
        return ray.get(self.execute_single_async(worker_index, func, *args, **kwargs))

    def remove_workers(self, worker_indexes: List[int]):
        """Removes the workers with the specified indexes.

        The removed workers will go out of scope and their actor processes
        will be terminated.

        Args:
            worker_indexes (List[int]): The indexes of the workers to remove.
        """
        new_workers = []
        for i in range(len(self.workers)):
            if i not in worker_indexes:
                new_workers.append(self.workers[i])
        self.workers = new_workers

    def add_workers(self, num_workers: int):
        """Adds ``num_workers`` to this WorkerGroup.

        Note: Adding workers when the cluster/placement group is at capacity
        may lead to undefined hanging behavior. If you are attempting to
        replace existing workers in the WorkerGroup, remove_workers() should
        be called first.

        Args:
            num_workers: The number of workers to add.
        """
        new_actors = []
        new_actor_metadata = []
        for _ in range(num_workers):
            actor = self._remote_cls.options(placement_group=self._placement_group).remote(*self._actor_cls_args, **self._actor_cls_kwargs)
            new_actors.append(actor)
            new_actor_metadata.append(actor._RayTrainWorker__execute.options(name='_RayTrainWorker__execute.construct_metadata').remote(construct_metadata))
        metadata = ray.get(new_actor_metadata)
        for i in range(len(new_actors)):
            self.workers.append(Worker(actor=new_actors[i], metadata=metadata[i]))

    def sort_workers_by_ip_and_gpu_id(self, _first_ip: Optional[str]=None):
        """Reorder the workers by their node ip and the lowest GPU id.

        This is useful for collocating workers on the same node.

        Example:
            Given workers with the following attributes:
                worker_0: ip=1, gpu_ids=[1]
                worker_1: ip=0, gpu_ids=[0]
                worker_2: ip=1, gpu_ids=[0]
                worker_3: ip=0, gpu_ids=[1]

            The function will perform the following steps:
                1. Group by node IP:
                    ip=0: worker_1, worker_3
                    ip=1: worker_0, worker_2

                2. Sort each group by GPU ID:
                    ip=0: worker_1 (gpu_id=0), worker_3 (gpu_id=1)
                    ip=1: worker_2 (gpu_id=0), worker_0 (gpu_id=1)

            Resulting in the order: [worker_1, worker_3, worker_2, worker_0]

        Args:
            _first_ip: The first IP to group by.
                Hack to avoid OOMs.
                This is just a temporary solution for Train loading entire checkpoints
                into memory by ensuring that the rank 0 worker is on the same node as
                trainable, thus allowing for lazy checkpoint transfer to be used.
                See https://github.com/ray-project/ray/issues/33073
                for more context.
                TODO remove this argument.
        """
        ip_to_workers = defaultdict(list)
        if _first_ip is not None:
            ip_to_workers[_first_ip] = []
        for worker in self.workers:
            ip_to_workers[worker.metadata.node_ip].append(worker)

        def get_lowest_gpu_id(worker) -> int:
            gpu_ids = worker.metadata.resource_ids.get('GPU', [])
            if not gpu_ids:
                return 0
            try:
                return min((int(gpu_id) for gpu_id in gpu_ids))
            except ValueError:
                return min(gpu_ids)
        for node_ip in ip_to_workers:
            ip_to_workers[node_ip].sort(key=get_lowest_gpu_id)
        sorted_workers = []
        for workers in ip_to_workers.values():
            sorted_workers.extend(workers)
        self.workers = sorted_workers

    def __len__(self):
        return len(self.workers)