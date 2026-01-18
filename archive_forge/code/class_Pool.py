import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
class Pool:
    """A pool of actor processes that is used to process tasks in parallel.

    Args:
        processes: number of actor processes to start in the pool. Defaults to
            the number of cores in the Ray cluster if one is already running,
            otherwise the number of cores on this machine.
        initializer: function to be run in each actor when it starts up.
        initargs: iterable of arguments to the initializer function.
        maxtasksperchild: maximum number of tasks to run in each actor process.
            After a process has executed this many tasks, it will be killed and
            replaced with a new one.
        ray_address: address of the Ray cluster to run on. If None, a new local
            Ray cluster will be started on this machine. Otherwise, this will
            be passed to `ray.init()` to connect to a running cluster. This may
            also be specified using the `RAY_ADDRESS` environment variable.
        ray_remote_args: arguments used to configure the Ray Actors making up
            the pool.
    """

    def __init__(self, processes: Optional[int]=None, initializer: Optional[Callable]=None, initargs: Optional[Iterable]=None, maxtasksperchild: Optional[int]=None, context: Any=None, ray_address: Optional[str]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        usage_lib.record_library_usage('util.multiprocessing.Pool')
        self._closed = False
        self._initializer = initializer
        self._initargs = initargs
        self._maxtasksperchild = maxtasksperchild or -1
        self._actor_deletion_ids = []
        self._registry: List[Tuple[Any, ray.ObjectRef]] = []
        self._registry_hashable: Dict[Hashable, ray.ObjectRef] = {}
        self._current_index = 0
        self._ray_remote_args = ray_remote_args or {}
        self._pool_actor = None
        if context and log_once('context_argument_warning'):
            logger.warning("The 'context' argument is not supported using ray. Please refer to the documentation for how to control ray initialization.")
        processes = self._init_ray(processes, ray_address)
        self._start_actor_pool(processes)

    def _init_ray(self, processes=None, ray_address=None):
        if not ray.is_initialized():
            if ray_address is None and (RAY_ADDRESS_ENV in os.environ or ray._private.utils.read_ray_address() is not None):
                ray.init()
            elif ray_address is not None:
                init_kwargs = {}
                if ray_address == 'local':
                    init_kwargs['num_cpus'] = processes
                ray.init(address=ray_address, **init_kwargs)
            else:
                ray.init(num_cpus=processes)
        ray_cpus = int(ray._private.state.cluster_resources()['CPU'])
        if processes is None:
            processes = ray_cpus
        if processes <= 0:
            raise ValueError('Processes in the pool must be >0.')
        if ray_cpus < processes:
            raise ValueError('Tried to start a pool with {} processes on an existing ray cluster, but there are only {} CPUs in the ray cluster.'.format(processes, ray_cpus))
        return processes

    def _start_actor_pool(self, processes):
        self._pool_actor = None
        self._actor_pool = [self._new_actor_entry() for _ in range(processes)]
        ray.get([actor.ping.remote() for actor, _ in self._actor_pool])

    def _wait_for_stopping_actors(self, timeout=None):
        if len(self._actor_deletion_ids) == 0:
            return
        if timeout is not None:
            timeout = float(timeout)
        _, deleting = ray.wait(self._actor_deletion_ids, num_returns=len(self._actor_deletion_ids), timeout=timeout)
        self._actor_deletion_ids = deleting

    def _stop_actor(self, actor):
        self._wait_for_stopping_actors(timeout=0.0)
        self._actor_deletion_ids.append(actor.__ray_terminate__.remote())

    def _new_actor_entry(self):
        if not self._pool_actor:
            self._pool_actor = PoolActor.options(**self._ray_remote_args)
        return (self._pool_actor.remote(self._initializer, self._initargs), 0)

    def _next_actor_index(self):
        if self._current_index == len(self._actor_pool) - 1:
            self._current_index = 0
        else:
            self._current_index += 1
        return self._current_index

    def _run_batch(self, actor_index, func, batch):
        actor, count = self._actor_pool[actor_index]
        object_ref = actor.run_batch.remote(func, batch)
        count += 1
        assert self._maxtasksperchild == -1 or count <= self._maxtasksperchild
        if count == self._maxtasksperchild:
            self._stop_actor(actor)
            actor, count = self._new_actor_entry()
        self._actor_pool[actor_index] = (actor, count)
        return object_ref

    def apply(self, func: Callable, args: Optional[Tuple]=None, kwargs: Optional[Dict]=None):
        """Run the given function on a random actor process and return the
        result synchronously.

        Args:
            func: function to run.
            args: optional arguments to the function.
            kwargs: optional keyword arguments to the function.

        Returns:
            The result.
        """
        return self.apply_async(func, args, kwargs).get()

    def apply_async(self, func: Callable, args: Optional[Tuple]=None, kwargs: Optional[Dict]=None, callback: Callable[[Any], None]=None, error_callback: Callable[[Exception], None]=None):
        """Run the given function on a random actor process and return an
        asynchronous interface to the result.

        Args:
            func: function to run.
            args: optional arguments to the function.
            kwargs: optional keyword arguments to the function.
            callback: callback to be executed on the result once it is finished
                only if it succeeds.
            error_callback: callback to be executed the result once it is
                finished only if the task errors. The exception raised by the
                task will be passed as the only argument to the callback.

        Returns:
            AsyncResult containing the result.
        """
        self._check_running()
        func = self._convert_to_ray_batched_calls_if_needed(func)
        object_ref = self._run_batch(self._next_actor_index(), func, [(args, kwargs)])
        return AsyncResult([object_ref], callback, error_callback, single_result=True)

    def _convert_to_ray_batched_calls_if_needed(self, func: Callable) -> Callable:
        """Convert joblib's BatchedCalls to RayBatchedCalls for ObjectRef caching.

        This converts joblib's BatchedCalls callable, which is a collection of
        functions with their args and kwargs to be ran sequentially in an
        Actor, to a RayBatchedCalls callable, which provides identical
        functionality in addition to a method which ensures that common
        args and kwargs are put into the object store just once, saving time
        and memory. That method is then ran.

        If func is not a BatchedCalls instance, it is returned without changes.

        The ObjectRefs are cached inside two registries (_registry and
        _registry_hashable), which are common for the entire Pool and are
        cleaned on close."""
        if RayBatchedCalls is None:
            return func
        orginal_func = func
        if isinstance(func, SafeFunction):
            func = func.func
        if isinstance(func, BatchedCalls):
            func = RayBatchedCalls(func.items, (func._backend, func._n_jobs), func._reducer_callback, func._pickle_cache)
            func.put_items_in_object_store(self._registry, self._registry_hashable)
        else:
            func = orginal_func
        return func

    def _calculate_chunksize(self, iterable):
        chunksize, extra = divmod(len(iterable), len(self._actor_pool) * 4)
        if extra:
            chunksize += 1
        return chunksize

    def _submit_chunk(self, func, iterator, chunksize, actor_index, unpack_args=False):
        chunk = []
        while len(chunk) < chunksize:
            try:
                args = next(iterator)
                if not unpack_args:
                    args = (args,)
                chunk.append((args, {}))
            except StopIteration:
                break
        assert len(chunk) > 0
        return self._run_batch(actor_index, func, chunk)

    def _chunk_and_run(self, func, iterable, chunksize=None, unpack_args=False):
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)
        if chunksize is None:
            chunksize = self._calculate_chunksize(iterable)
        iterator = iter(iterable)
        chunk_object_refs = []
        while len(chunk_object_refs) * chunksize < len(iterable):
            actor_index = len(chunk_object_refs) % len(self._actor_pool)
            chunk_object_refs.append(self._submit_chunk(func, iterator, chunksize, actor_index, unpack_args=unpack_args))
        return chunk_object_refs

    def _map_async(self, func, iterable, chunksize=None, unpack_args=False, callback=None, error_callback=None):
        self._check_running()
        object_refs = self._chunk_and_run(func, iterable, chunksize=chunksize, unpack_args=unpack_args)
        return AsyncResult(object_refs, callback, error_callback)

    def map(self, func: Callable, iterable: Iterable, chunksize: Optional[int]=None):
        """Run the given function on each element in the iterable round-robin
        on the actor processes and return the results synchronously.

        Args:
            func: function to run.
            iterable: iterable of objects to be passed as the sole argument to
                func.
            chunksize: number of tasks to submit as a batch to each actor
                process. If unspecified, a suitable chunksize will be chosen.

        Returns:
            A list of results.
        """
        return self._map_async(func, iterable, chunksize=chunksize, unpack_args=False).get()

    def map_async(self, func: Callable, iterable: Iterable, chunksize: Optional[int]=None, callback: Callable[[List], None]=None, error_callback: Callable[[Exception], None]=None):
        """Run the given function on each element in the iterable round-robin
        on the actor processes and return an asynchronous interface to the
        results.

        Args:
            func: function to run.
            iterable: iterable of objects to be passed as the only argument to
                func.
            chunksize: number of tasks to submit as a batch to each actor
                process. If unspecified, a suitable chunksize will be chosen.
            callback: Will only be called if none of the results were errors,
                and will only be called once after all results are finished.
                A Python List of all the finished results will be passed as the
                only argument to the callback.
            error_callback: callback executed on the first errored result.
                The Exception raised by the task will be passed as the only
                argument to the callback.

        Returns:
            AsyncResult
        """
        return self._map_async(func, iterable, chunksize=chunksize, unpack_args=False, callback=callback, error_callback=error_callback)

    def starmap(self, func, iterable, chunksize=None):
        """Same as `map`, but unpacks each element of the iterable as the
        arguments to func like: [func(*args) for args in iterable].
        """
        return self._map_async(func, iterable, chunksize=chunksize, unpack_args=True).get()

    def starmap_async(self, func: Callable, iterable: Iterable, callback: Callable[[List], None]=None, error_callback: Callable[[Exception], None]=None):
        """Same as `map_async`, but unpacks each element of the iterable as the
        arguments to func like: [func(*args) for args in iterable].
        """
        return self._map_async(func, iterable, unpack_args=True, callback=callback, error_callback=error_callback)

    def imap(self, func: Callable, iterable: Iterable, chunksize: Optional[int]=1):
        """Same as `map`, but only submits one batch of tasks to each actor
        process at a time.

        This can be useful if the iterable of arguments is very large or each
        task's arguments consumes a large amount of resources.

        The results are returned in the order corresponding to their arguments
        in the iterable.

        Returns:
            OrderedIMapIterator
        """
        self._check_running()
        return OrderedIMapIterator(self, func, iterable, chunksize=chunksize)

    def imap_unordered(self, func: Callable, iterable: Iterable, chunksize: Optional[int]=1):
        """Same as `map`, but only submits one batch of tasks to each actor
        process at a time.

        This can be useful if the iterable of arguments is very large or each
        task's arguments consumes a large amount of resources.

        The results are returned in the order that they finish.

        Returns:
            UnorderedIMapIterator
        """
        self._check_running()
        return UnorderedIMapIterator(self, func, iterable, chunksize=chunksize)

    def _check_running(self):
        if self._closed:
            raise ValueError('Pool not running')

    def __enter__(self):
        self._check_running()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def close(self):
        """Close the pool.

        Prevents any more tasks from being submitted on the pool but allows
        outstanding work to finish.
        """
        self._registry.clear()
        self._registry_hashable.clear()
        for actor, _ in self._actor_pool:
            self._stop_actor(actor)
        self._closed = True
        gc.collect()

    def terminate(self):
        """Close the pool.

        Prevents any more tasks from being submitted on the pool and stops
        outstanding work.
        """
        if not self._closed:
            self.close()
        for actor, _ in self._actor_pool:
            ray.kill(actor)

    def join(self):
        """Wait for the actors in a closed pool to exit.

        If the pool was closed using `close`, this will return once all
        outstanding work is completed.

        If the pool was closed using `terminate`, this will return quickly.
        """
        if not self._closed:
            raise ValueError('Pool is still running')
        self._wait_for_stopping_actors()