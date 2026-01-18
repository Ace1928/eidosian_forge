import asyncio
import os
from types import FunctionType
from typing import Sequence
import ray
from ray.util.client.common import ClientObjectRef
from modin.error_message import ErrorMessage
class RayWrapper:
    """Mixin that provides means of running functions remotely and getting local results."""
    _func_cache = {}

    @classmethod
    def deploy(cls, func, f_args=None, f_kwargs=None, num_returns=1):
        """
        Run local `func` remotely.

        Parameters
        ----------
        func : callable or ray.ObjectID
            The function to perform.
        f_args : list or tuple, optional
            Positional arguments to pass to ``func``.
        f_kwargs : dict, optional
            Keyword arguments to pass to ``func``.
        num_returns : int, default: 1
            Amount of return values expected from `func`.

        Returns
        -------
        ray.ObjectRef or list
            Ray identifier of the result being put to Plasma store.
        """
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        return _deploy_ray_func.options(num_returns=num_returns).remote(func, *args, **kwargs)

    @classmethod
    def is_future(cls, item):
        """
        Check if the item is a Future.

        Parameters
        ----------
        item : ray.ObjectID or object
            Future or object to check.

        Returns
        -------
        boolean
            If the value is a future.
        """
        return isinstance(item, ObjectRefTypes)

    @classmethod
    def materialize(cls, obj_id):
        """
        Get the value of object from the Plasma store.

        Parameters
        ----------
        obj_id : ray.ObjectID
            Ray object identifier to get the value by.

        Returns
        -------
        object
            Whatever was identified by `obj_id`.
        """
        if isinstance(obj_id, MaterializationHook):
            obj = obj_id.pre_materialize()
            return obj_id.post_materialize(ray.get(obj)) if isinstance(obj, RayObjectRefTypes) else obj
        if not isinstance(obj_id, Sequence):
            return ray.get(obj_id) if isinstance(obj_id, RayObjectRefTypes) else obj_id
        if all((isinstance(obj, RayObjectRefTypes) for obj in obj_id)):
            return ray.get(obj_id)
        ids = {}
        result = []
        for obj in obj_id:
            if not isinstance(obj, ObjectRefTypes):
                result.append(obj)
                continue
            if isinstance(obj, MaterializationHook):
                oid = obj.pre_materialize()
                if isinstance(oid, RayObjectRefTypes):
                    hook = obj
                    obj = oid
                else:
                    result.append(oid)
                    continue
            else:
                hook = None
            idx = ids.get(obj, None)
            if idx is None:
                ids[obj] = idx = len(ids)
            if hook is None:
                result.append(obj)
            else:
                hook._materialized_idx = idx
                result.append(hook)
        if len(ids) == 0:
            return result
        materialized = ray.get(list(ids.keys()))
        for i in range(len(result)):
            if isinstance((obj := result[i]), ObjectRefTypes):
                if isinstance(obj, MaterializationHook):
                    result[i] = obj.post_materialize(materialized[obj._materialized_idx])
                else:
                    result[i] = materialized[ids[obj]]
        return result

    @classmethod
    def put(cls, data, **kwargs):
        """
        Store an object in the object store.

        Parameters
        ----------
        data : object
            The Python object to be stored.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        ray.ObjectID
            Ray object identifier to get the value by.
        """
        if isinstance(data, FunctionType):
            qname = data.__qualname__
            if '<locals>' not in qname and '<lambda>' not in qname:
                ref = cls._func_cache.get(data, None)
                if ref is None:
                    if len(cls._func_cache) < 1024:
                        ref = ray.put(data)
                        cls._func_cache[data] = ref
                    else:
                        msg = 'To many functions in the RayWrapper cache!'
                        assert 'MODIN_GITHUB_CI' not in os.environ, msg
                        ErrorMessage.warn(msg)
                return ref
        return ray.put(data, **kwargs)

    @classmethod
    def wait(cls, obj_ids, num_returns=None):
        """
        Wait on the objects without materializing them (blocking operation).

        ``ray.wait`` assumes a list of unique object references: see
        https://github.com/modin-project/modin/issues/5045

        Parameters
        ----------
        obj_ids : list, scalar
        num_returns : int, optional
        """
        if not isinstance(obj_ids, Sequence):
            obj_ids = list(obj_ids)
        ids = set()
        for obj in obj_ids:
            if isinstance(obj, MaterializationHook):
                obj = obj.pre_materialize()
            if isinstance(obj, RayObjectRefTypes):
                ids.add(obj)
        if (num_ids := len(ids)):
            ray.wait(list(ids), num_returns=num_returns or num_ids)