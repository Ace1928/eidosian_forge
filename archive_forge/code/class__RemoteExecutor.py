from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
class _RemoteExecutor:
    """Remote functions for DeferredExecution."""

    @staticmethod
    def exec_func(fn: Callable, obj: Any, args: Tuple, kwargs: Dict) -> Any:
        """
        Execute the specified function.

        Parameters
        ----------
        fn : Callable
        obj : Any
        args : Tuple
        kwargs : dict

        Returns
        -------
        Any
        """
        try:
            try:
                return fn(obj, *args, **kwargs)
            except ValueError as err:
                if isinstance(obj, (pandas.DataFrame, pandas.Series)):
                    return fn(obj.copy(), *args, **kwargs)
                else:
                    raise err
        except Exception as err:
            get_logger().error(f'{err}. fn={fn}, obj={obj}, args={args}, kwargs={kwargs}')
            raise err

    @classmethod
    def construct(cls, num_returns: int, args: Tuple):
        """
        Construct and execute the specified chain.

        This function is called in a worker process. The last value, returned by
        this generator, is the meta list, containing the objects lengths and widths
        and the worker ip address, as the last value in the list.

        Parameters
        ----------
        num_returns : int
        args : tuple

        Yields
        ------
        Any
            The execution results and the MetaList as the last value.
        """
        chain = list(reversed(args))
        meta = []
        try:
            stack = [cls.construct_chain(chain, {}, meta, None)]
            while stack:
                try:
                    gen = stack.pop()
                    obj = next(gen)
                    stack.append(gen)
                    if isinstance(obj, Generator):
                        stack.append(obj)
                    else:
                        yield obj
                except StopIteration:
                    pass
        except Exception as err:
            get_logger().error(f'{err}. args={args}, chain={list(reversed(chain))}')
            raise err
        meta.append(get_node_ip_address())
        yield meta

    @classmethod
    def construct_chain(cls, chain: List, refs: Dict[int, Any], meta: List, lst: Optional[List]):
        """
        Construct the chain and execute it one by one.

        Parameters
        ----------
        chain : list
            A flat list containing the execution tree, deconstructed by
            ``DeferredExecution._deconstruct()``.
        refs : dict
            If an execution result is required for multiple chains, the
            reference to this result is saved in this dict.
        meta : list
            The lengths of the returned objects are added to this list.
        lst : list
            If specified, the execution result is added to this list.
            This is used when a chain is passed as an argument to a
            DeferredExecution task.

        Yields
        ------
        Any
            Either the ``construct_list()`` generator or the execution results.
        """
        pop = chain.pop
        tg_e = _Tag.END
        obj = pop()
        if obj is _Tag.REF:
            obj = refs[pop()]
        elif obj is _Tag.LIST:
            obj = []
            yield cls.construct_list(obj, chain, refs, meta)
        while chain:
            fn = pop()
            if fn == tg_e:
                lst.append(obj)
                break
            if (args_len := pop()) >= 0:
                if args_len == 0:
                    args = []
                else:
                    args = chain[-args_len:]
                    del chain[-args_len:]
                    args.reverse()
            else:
                args = []
                yield cls.construct_list(args, chain, refs, meta)
            if (args_len := pop()) >= 0:
                kwargs = {pop(): pop() for _ in range(args_len)}
            else:
                values = []
                yield cls.construct_list(values, chain, refs, meta)
                kwargs = {pop(): v for v in values}
            obj = cls.exec_func(fn, obj, args, kwargs)
            if (ref := pop()):
                refs[ref] = obj
            if (num_returns := pop()) == 0:
                continue
            itr = iter([obj] if num_returns == 1 else obj)
            for _ in range(num_returns):
                obj = next(itr)
                meta.append(len(obj) if hasattr(obj, '__len__') else 0)
                meta.append(len(obj.columns) if hasattr(obj, 'columns') else 0)
                yield obj

    @classmethod
    def construct_list(cls, lst: List, chain: List, refs: Dict[int, Any], meta: List):
        """
        Construct the list.

        Parameters
        ----------
        lst : list
        chain : list
        refs : dict
        meta : list

        Yields
        ------
        Any
            Either ``construct_chain()`` or ``construct_list()`` generator.
        """
        pop = chain.pop
        lst_append = lst.append
        while True:
            obj = pop()
            if isinstance(obj, _Tag):
                if obj == _Tag.END:
                    break
                elif obj == _Tag.CHAIN:
                    yield cls.construct_chain(chain, refs, meta, lst)
                elif obj == _Tag.LIST:
                    lst_append([])
                    yield cls.construct_list(lst[-1], chain, refs, meta)
                elif obj is _Tag.REF:
                    lst_append(refs[pop()])
                else:
                    raise ValueError(f'Unexpected tag {obj}')
            else:
                lst_append(obj)

    def __reduce__(self):
        """
        Use a single instance on deserialization.

        Returns
        -------
        str
            Returns the ``_REMOTE_EXEC`` attribute name.
        """
        return '_REMOTE_EXEC'