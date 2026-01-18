import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
@Deprecated
class ParallelIterator(Generic[T]):
    """A parallel iterator over a set of remote actors.

    This can be used to iterate over a fixed set of task results
    (like an actor pool), or a stream of data (e.g., a fixed range of numbers,
    an infinite stream of RLlib rollout results).

    This class is **serializable** and can be passed to other remote
    tasks and actors. However, each shard should be read from at most one
    process at a time.

    Examples:
        >>> # Applying a function over items in parallel.
        >>> it = ray.util.iter.from_items([1, 2, 3], num_shards=2)
        ... <__main__.ParallelIterator object>
        >>> it = it.for_each(lambda x: x * 2).gather_sync()
        ... <__main__.LocalIterator object>
        >>> print(list(it))
        ... [2, 4, 6]

        >>> # Creating from generators.
        >>> it = ray.util.iter.from_iterators([range(3), range(3)])
        ... <__main__.ParallelIterator object>
        >>> print(list(it.gather_sync()))
        ... [0, 0, 1, 1, 2, 2]

        >>> # Accessing the individual shards of an iterator.
        >>> it = ray.util.iter.from_range(10, num_shards=2)
        ... <__main__.ParallelIterator object>
        >>> it0 = it.get_shard(0)
        ... <__main__.LocalIterator object>
        >>> print(list(it0))
        ... [0, 1, 2, 3, 4]
        >>> it1 = it.get_shard(1)
        ... <__main__.LocalIterator object>
        >>> print(list(it1))
        ... [5, 6, 7, 8, 9]

        >>> # Gathering results from actors synchronously in parallel.
        >>> it = ray.util.iter.from_actors(workers)
        ... <__main__.ParallelIterator object>
        >>> it = it.batch_across_shards()
        ... <__main__.LocalIterator object>
        >>> print(next(it))
        ... [worker_1_result_1, worker_2_result_1]
        >>> print(next(it))
        ... [worker_1_result_2, worker_2_result_2]
    """

    def __init__(self, actor_sets: List['_ActorSet'], name: str, parent_iterators: List['ParallelIterator[Any]']):
        """Create a parallel iterator (this is an internal function)."""
        self.actor_sets = actor_sets
        self.name = name
        self.parent_iterators = parent_iterators

    def __iter__(self):
        raise TypeError('You must use it.gather_sync() or it.gather_async() to iterate over the results of a ParallelIterator.')

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'ParallelIterator[{self.name}]'

    def _with_transform(self, local_it_fn, name):
        """Helper function to create new Parallel Iterator"""
        return ParallelIterator([a.with_transform(local_it_fn) for a in self.actor_sets], name=self.name + name, parent_iterators=self.parent_iterators)

    def transform(self, fn: Callable[[Iterable[T]], Iterable[U]]) -> 'ParallelIterator[U]':
        """Remotely transform the iterator.

        This is advanced version of for_each that allows you to apply arbitrary
        generator transformations over the iterator. Prefer to use .for_each()
        when possible for simplicity.

        Args:
            fn: function to use to transform the iterator. The function
                should pass through instances of _NextValueNotReady that appear
                in its input iterator. Note that this function is only called
                **once** over the input iterator.

        Returns:
            ParallelIterator[U]: a parallel iterator.

        Examples:
            >>> def f(it):
            ...     for x in it:
            ...         if x % 2 == 0:
            ...            yield x
            >>> from_range(10, 1).transform(f).gather_sync().take(5)
            ... [0, 2, 4, 6, 8]
        """
        return self._with_transform(lambda local_it: local_it.transform(fn), '.transform()')

    def for_each(self, fn: Callable[[T], U], max_concurrency=1, resources=None) -> 'ParallelIterator[U]':
        """Remotely apply fn to each item in this iterator.

        If `max_concurrency` == 1 then `fn` will be executed serially by each
        shards

        `max_concurrency` should be used to achieve a high degree of
        parallelism without the overhead of increasing the number of shards
        (which are actor based). If `max_concurrency` is not 1, this function
        provides no semantic guarantees on the output order.
        Results will be returned as soon as they are ready.

        A performance note: When executing concurrently, this function
        maintains its own internal buffer. If `num_async` is `n` and
        max_concur is `k` then the total number of buffered objects could be up
        to `n + k - 1`

        Args:
            fn: function to apply to each item.
            max_concurrency: max number of concurrent calls to fn per
                shard. If 0, then apply all operations concurrently.
            resources: resources that the function requires to execute.
                This has the same default as `ray.remote` and is only used
                when `max_concurrency > 1`.

        Returns:
            ParallelIterator[U]: a parallel iterator whose elements have `fn`
            applied.

        Examples:
            >>> next(from_range(4).for_each(
                        lambda x: x * 2,
                        max_concur=2,
                        resources={"num_cpus": 0.1}).gather_sync()
                )
            ... [0, 2, 4, 8]

        """
        assert max_concurrency >= 0, 'max_concurrency must be non-negative.'
        return self._with_transform(lambda local_it: local_it.for_each(fn, max_concurrency, resources), '.for_each()')

    def filter(self, fn: Callable[[T], bool]) -> 'ParallelIterator[T]':
        """Remotely filter items from this iterator.

        Args:
            fn: returns False for items to drop from the iterator.

        Examples:
            >>> it = from_items([0, 1, 2]).filter(lambda x: x > 0)
            >>> next(it.gather_sync())
            ... [1, 2]
        """
        return self._with_transform(lambda local_it: local_it.filter(fn), '.filter()')

    def batch(self, n: int) -> 'ParallelIterator[List[T]]':
        """Remotely batch together items in this iterator.

        Args:
            n: Number of items to batch together.

        Examples:
            >>> next(from_range(10, 1).batch(4).gather_sync())
            ... [0, 1, 2, 3]
        """
        return self._with_transform(lambda local_it: local_it.batch(n), f'.batch({n})')

    def flatten(self) -> 'ParallelIterator[T[0]]':
        """Flatten batches of items into individual items.

        Examples:
            >>> next(from_range(10, 1).batch(4).flatten())
            ... 0
        """
        return self._with_transform(lambda local_it: local_it.flatten(), '.flatten()')

    def combine(self, fn: Callable[[T], List[U]]) -> 'ParallelIterator[U]':
        """Transform and then combine items horizontally.

        This is the equivalent of for_each(fn).flatten() (flat map).
        """
        it = self.for_each(fn).flatten()
        it.name = self.name + '.combine()'
        return it

    def local_shuffle(self, shuffle_buffer_size: int, seed: int=None) -> 'ParallelIterator[T]':
        """Remotely shuffle items of each shard independently

        Args:
            shuffle_buffer_size: The algorithm fills a buffer with
                shuffle_buffer_size elements and randomly samples elements from
                this buffer, replacing the selected elements with new elements.
                For perfect shuffling, this argument should be greater than or
                equal to the largest iterator size.
            seed: Seed to use for
                randomness. Default value is None.

        Returns:
            A ParallelIterator with a local shuffle applied on the base
            iterator

        Examples:
            >>> it = from_range(10, 1).local_shuffle(shuffle_buffer_size=2)
            >>> it = it.gather_sync()
            >>> next(it)
            0
            >>> next(it)
            2
            >>> next(it)
            3
            >>> next(it)
            1
        """
        return self._with_transform(lambda local_it: local_it.shuffle(shuffle_buffer_size, seed), '.local_shuffle(shuffle_buffer_size={}, seed={})'.format(shuffle_buffer_size, str(seed) if seed is not None else 'None'))

    def repartition(self, num_partitions: int, batch_ms: int=0) -> 'ParallelIterator[T]':
        """Returns a new ParallelIterator instance with num_partitions shards.

        The new iterator contains the same data in this instance except with
        num_partitions shards. The data is split in round-robin fashion for
        the new ParallelIterator.

        Args:
            num_partitions: The number of shards to use for the new
                ParallelIterator
            batch_ms: Batches items for batch_ms milliseconds
                on each shard before retrieving it.
                Increasing batch_ms increases latency but improves throughput.

        Returns:
            A ParallelIterator with num_partitions number of shards and the
            data of this ParallelIterator split round-robin among the new
            number of shards.

        Examples:
            >>> it = from_range(8, 2)
            >>> it = it.repartition(3)
            >>> list(it.get_shard(0))
            [0, 4, 3, 7]
            >>> list(it.get_shard(1))
            [1, 5]
            >>> list(it.get_shard(2))
            [2, 6]
        """
        all_actors = []
        for actor_set in self.actor_sets:
            actor_set.init_actors()
            all_actors.extend(actor_set.actors)

        def base_iterator(num_partitions, partition_index, timeout=None):
            futures = {}
            for a in all_actors:
                futures[a.par_iter_slice_batch.remote(step=num_partitions, start=partition_index, batch_ms=batch_ms)] = a
            while futures:
                pending = list(futures)
                if timeout is None:
                    ready, _ = ray.wait(pending, num_returns=len(pending), timeout=0)
                    if not ready:
                        ready, _ = ray.wait(pending, num_returns=1)
                else:
                    ready, _ = ray.wait(pending, num_returns=len(pending), timeout=timeout)
                for obj_ref in ready:
                    actor = futures.pop(obj_ref)
                    try:
                        batch = ray.get(obj_ref)
                        futures[actor.par_iter_slice_batch.remote(step=num_partitions, start=partition_index, batch_ms=batch_ms)] = actor
                        for item in batch:
                            yield item
                    except StopIteration:
                        pass
                if timeout is not None:
                    yield _NextValueNotReady()

        def make_gen_i(i):
            return lambda: base_iterator(num_partitions, i)
        name = self.name + f'.repartition[num_partitions={num_partitions}]'
        generators = [make_gen_i(s) for s in range(num_partitions)]
        worker_cls = ray.remote(ParallelIteratorWorker)
        actors = [worker_cls.remote(g, repeat=False) for g in generators]
        return ParallelIterator([_ActorSet(actors, [])], name, parent_iterators=[self])

    def gather_sync(self) -> 'LocalIterator[T]':
        """Returns a local iterable for synchronous iteration.

        New items will be fetched from the shards on-demand as the iterator
        is stepped through.

        This is the equivalent of batch_across_shards().flatten().

        Examples:
            >>> it = from_range(100, 1).gather_sync()
            >>> next(it)
            ... 0
            >>> next(it)
            ... 1
            >>> next(it)
            ... 2
        """
        it = self.batch_across_shards().flatten()
        it.name = f'{self}.gather_sync()'
        return it

    def batch_across_shards(self) -> 'LocalIterator[List[T]]':
        """Iterate over the results of multiple shards in parallel.

        Examples:
            >>> it = from_iterators([range(3), range(3)])
            >>> next(it.batch_across_shards())
            ... [0, 0]
        """

        def base_iterator(timeout=None):
            active = []
            for actor_set in self.actor_sets:
                actor_set.init_actors()
                active.extend(actor_set.actors)
            futures = [a.par_iter_next.remote() for a in active]
            while active:
                try:
                    yield ray.get(futures, timeout=timeout)
                    futures = [a.par_iter_next.remote() for a in active]
                    if timeout is not None:
                        yield _NextValueNotReady()
                except TimeoutError:
                    yield _NextValueNotReady()
                except StopIteration:
                    results = []
                    for a, f in zip(list(active), futures):
                        try:
                            results.append(ray.get(f))
                        except StopIteration:
                            active.remove(a)
                    if results:
                        yield results
                    futures = [a.par_iter_next.remote() for a in active]
        name = f'{self}.batch_across_shards()'
        return LocalIterator(base_iterator, SharedMetrics(), name=name)

    def gather_async(self, batch_ms=0, num_async=1) -> 'LocalIterator[T]':
        """Returns a local iterable for asynchronous iteration.

        New items will be fetched from the shards asynchronously as soon as
        the previous one is computed. Items arrive in non-deterministic order.

        Arguments:
            batch_ms: Batches items for batch_ms milliseconds
                on each shard before retrieving it.
                Increasing batch_ms increases latency but improves throughput.
                If this value is 0, then items are returned immediately.
            num_async: The max number of async requests in flight
                per actor. Increasing this improves the amount of pipeline
                parallelism in the iterator.

        Examples:
            >>> it = from_range(100, 1).gather_async()
            >>> next(it)
            ... 3
            >>> next(it)
            ... 0
            >>> next(it)
            ... 1
        """
        if num_async < 1:
            raise ValueError('queue depth must be positive')
        if batch_ms < 0:
            raise ValueError('batch time must be positive')
        local_iter = None

        def base_iterator(timeout=None):
            all_actors = []
            for actor_set in self.actor_sets:
                actor_set.init_actors()
                all_actors.extend(actor_set.actors)
            futures = {}
            for _ in range(num_async):
                for a in all_actors:
                    futures[a.par_iter_next_batch.remote(batch_ms)] = a
            while futures:
                pending = list(futures)
                if timeout is None:
                    ready, _ = ray.wait(pending, num_returns=len(pending), timeout=0)
                    if not ready:
                        ready, _ = ray.wait(pending, num_returns=1)
                else:
                    ready, _ = ray.wait(pending, num_returns=len(pending), timeout=timeout)
                for obj_ref in ready:
                    actor = futures.pop(obj_ref)
                    try:
                        local_iter.shared_metrics.get().current_actor = actor
                        batch = ray.get(obj_ref)
                        futures[actor.par_iter_next_batch.remote(batch_ms)] = actor
                        for item in batch:
                            yield item
                    except StopIteration:
                        pass
                if timeout is not None:
                    yield _NextValueNotReady()
        name = f'{self}.gather_async()'
        local_iter = LocalIterator(base_iterator, SharedMetrics(), name=name)
        return local_iter

    def take(self, n: int) -> List[T]:
        """Return up to the first n items from this iterator."""
        return self.gather_sync().take(n)

    def show(self, n: int=20):
        """Print up to the first n items from this iterator."""
        return self.gather_sync().show(n)

    def union(self, other: 'ParallelIterator[T]') -> 'ParallelIterator[T]':
        """Return an iterator that is the union of this and the other."""
        if not isinstance(other, ParallelIterator):
            raise TypeError(f'other must be of type ParallelIterator, got {type(other)}')
        actor_sets = []
        actor_sets.extend(self.actor_sets)
        actor_sets.extend(other.actor_sets)
        return ParallelIterator(actor_sets, f'ParallelUnion[{self}, {other}]', parent_iterators=self.parent_iterators + other.parent_iterators)

    def select_shards(self, shards_to_keep: List[int]) -> 'ParallelIterator[T]':
        """Return a child iterator that only iterates over given shards.

        It is the user's responsibility to ensure child iterators are operating
        over disjoint sub-sets of this iterator's shards.
        """
        if len(self.actor_sets) > 1:
            raise ValueError('select_shards() is not allowed after union()')
        if len(shards_to_keep) == 0:
            raise ValueError('at least one shard must be selected')
        old_actor_set = self.actor_sets[0]
        new_actors = [a for i, a in enumerate(old_actor_set.actors) if i in shards_to_keep]
        assert len(new_actors) == len(shards_to_keep), 'Invalid actor index'
        new_actor_set = _ActorSet(new_actors, old_actor_set.transforms)
        return ParallelIterator([new_actor_set], f'{self}.select_shards({len(shards_to_keep)} total)', parent_iterators=self.parent_iterators)

    def num_shards(self) -> int:
        """Return the number of worker actors backing this iterator."""
        return sum((len(a.actors) for a in self.actor_sets))

    def shards(self) -> List['LocalIterator[T]']:
        """Return the list of all shards."""
        return [self.get_shard(i) for i in range(self.num_shards())]

    def get_shard(self, shard_index: int, batch_ms: int=0, num_async: int=1) -> 'LocalIterator[T]':
        """Return a local iterator for the given shard.

        The iterator is guaranteed to be serializable and can be passed to
        remote tasks or actors.

        Arguments:
            shard_index: Index of the shard to gather.
            batch_ms: Batches items for batch_ms milliseconds
                before retrieving it.
                Increasing batch_ms increases latency but improves throughput.
                If this value is 0, then items are returned immediately.
            num_async: The max number of requests in flight.
                Increasing this improves the amount of pipeline
                parallelism in the iterator.
        """
        if num_async < 1:
            raise ValueError('num async must be positive')
        if batch_ms < 0:
            raise ValueError('batch time must be positive')
        a, t = (None, None)
        i = shard_index
        for actor_set in self.actor_sets:
            if i < len(actor_set.actors):
                a = actor_set.actors[i]
                t = actor_set.transforms
                break
            else:
                i -= len(actor_set.actors)
        if a is None:
            raise ValueError('Shard index out of range', shard_index, self.num_shards())

        def base_iterator(timeout=None):
            queue = collections.deque()
            ray.get(a.par_iter_init.remote(t))
            for _ in range(num_async):
                queue.append(a.par_iter_next_batch.remote(batch_ms))
            while True:
                try:
                    batch = ray.get(queue.popleft(), timeout=timeout)
                    queue.append(a.par_iter_next_batch.remote(batch_ms))
                    for item in batch:
                        yield item
                    if timeout is not None:
                        yield _NextValueNotReady()
                except TimeoutError:
                    yield _NextValueNotReady()
                except StopIteration:
                    break
        name = self.name + f'.shard[{shard_index}]'
        return LocalIterator(base_iterator, SharedMetrics(), name=name)