import itertools
from functools import update_wrapper
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from .entry_points import load_entry_point
class ConditionalDispatcher:
    """A conditional function dispatcher based on custom matching functions.
    This is a more general solution compared to
    ``functools.singledispatch``. You can write arbitrary matching functions according
    to all the inputs of the function.

    .. note::

        Please use the decorators :func:`.conditional_dispatcher` and
        :func:`.conditional_broadcaster` instead of directly using this class.

    :param default_func: the parent function that will dispatch the execution
        based on matching functions
    :param entry_point: the entry point to preload children functions,
        defaults to None
    """

    def __init__(self, default_func: Callable[..., Any], is_broadcast: bool, entry_point: Optional[str]=None):
        self._func = default_func
        self._funcs: List[Tuple[float, int, Callable[..., bool], Callable[..., Any]]] = []
        self._entry_point = entry_point
        self._is_broadcast = is_broadcast
        update_wrapper(self, default_func)

    def __getstate__(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k in ['_func', '_funcs', '_entry_point', '_is_broadcast']}

    def __setstate__(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            setattr(self, k, v)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """The abstract method to mimic the function call"""
        if self._is_broadcast:
            return list(self.run(*args, **kwds))
        return self.run_top(*args, **kwds)

    def run(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        """Execute all matching children functions as a generator.

        .. note::

            Only when there is matching functions, the default implementation
            will be invoked.
        """
        self._preload()
        has_return = False
        for f in self._funcs:
            if self._match(f[2], *args, **kwargs):
                yield f[3](*args, **kwargs)
                has_return = True
        if not has_return:
            yield self._func(*args, **kwargs)

    def run_top(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the first matching child function

        :return: the return of the child function
        """
        return list(itertools.islice(self.run(*args, **kwargs), 1))[0]

    def register(self, func: Callable[..., Any], matcher: Callable[..., bool], priority: float=1.0) -> None:
        """Register a child function with matcher and priority.

        .. note::

            The order to be matched is determined by both the priority
            and the order of registration.

            * The default priority is 1.0
            * Children with higher priority values will be matched earlier
            * When ``priority>0`` then later registrations will be matched earlier
            * When ``priority<=0`` then earlier registrations will be matched earlier

            So if you want to 'overwrite' the existed matches, set priority to be
            greater than 1.0. If you want to 'ignore' the current if there are other
            matches, set priority to 0.0.

        :param func: a child function to be used when matching
        :param matcher: a function determines whether it is a match
            based on the same input as the parent function
        :param priority: it determines the order to be matched,
            **higher value means higher priority**, defaults to 1.0
        """
        n = len(self._funcs)
        self._funcs.append((-priority, n if priority <= 0.0 else -n, matcher, func))
        self._funcs.sort()

    def candidate(self, matcher: Callable[..., bool], priority: float=1.0) -> Callable:
        """A decorator to register a child function with matcher and priority.

        .. note::

            The order to be matched is determined by both the priority
            and the order of registration.

            * The default priority is 1.0
            * Children with higher priority values will be matched earlier
            * When ``priority>0`` then later registrations will be matched earlier
            * When ``priority<=0`` then earlier registrations will be matched earlier

            So if you want to 'overwrite' the existed matches, set priority to be
            greater than 1.0. If you want to 'ignore' the current if there are other
            matches, set priority to 0.0.

        .. seealso::

            Please see examples in :func:`.conditional_dispatcher` and
            :func:`.conditional_broadcaster`.

        :param matcher: a function determines whether it is a match
            based on the same input as the parent function
        :param priority: it determines the order to be matched,
            **higher value means higher priority**, defaults to 1.0
        """

        def _run(_func: Callable[..., Any]):
            self.register(_func, matcher=matcher, priority=priority)
            return _func
        return _run

    def _preload(self) -> None:
        if self._entry_point is not None:
            load_entry_point(self._entry_point)

    def _match(self, m: Callable[..., bool], *args: Any, **kwargs: Any) -> bool:
        try:
            return m(*args, **kwargs)
        except Exception:
            return False