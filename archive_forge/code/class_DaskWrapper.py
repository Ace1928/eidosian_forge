from collections import UserDict
from dask.distributed import wait
from distributed import Future
from distributed.client import default_client
class DaskWrapper:
    """The class responsible for execution of remote operations."""

    @classmethod
    def deploy(cls, func, f_args=None, f_kwargs=None, num_returns=1, pure=True):
        """
        Deploy a function in a worker process.

        Parameters
        ----------
        func : callable or distributed.Future
            Function to be deployed in a worker process.
        f_args : list or tuple, optional
            Positional arguments to pass to ``func``.
        f_kwargs : dict, optional
            Keyword arguments to pass to ``func``.
        num_returns : int, default: 1
            The number of returned objects.
        pure : bool, default: True
            Whether or not `func` is pure. See `Client.submit` for details.

        Returns
        -------
        list
            The result of ``func`` split into parts in accordance with ``num_returns``.
        """
        client = default_client()
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        if callable(func):
            remote_task_future = client.submit(func, *args, pure=pure, **kwargs)
        else:
            remote_task_future = client.submit(_deploy_dask_func, func, *args, pure=pure, **kwargs)
        if num_returns != 1:
            return [client.submit(lambda tup, i: tup[i], remote_task_future, i) for i in range(num_returns)]
        return remote_task_future

    @classmethod
    def is_future(cls, item):
        """
        Check if the item is a Future.

        Parameters
        ----------
        item : distributed.Future or object
            Future or object to check.

        Returns
        -------
        boolean
            If the value is a future.
        """
        return isinstance(item, Future)

    @classmethod
    def materialize(cls, future):
        """
        Materialize data matching `future` object.

        Parameters
        ----------
        future : distributed.Future or list
            Future object of list of future objects whereby data needs to be materialized.

        Returns
        -------
        Any
            An object(s) from the distributed memory.
        """
        client = default_client()
        return client.gather(future)

    @classmethod
    def put(cls, data, **kwargs):
        """
        Put data into distributed memory.

        Parameters
        ----------
        data : list, dict, or object
            Data to scatter out to workers. Output type matches input type.
        **kwargs : dict
            Additional keyword arguments to be passed in `Client.scatter`.

        Returns
        -------
        List, dict, iterator, or queue of futures matching the type of input.
        """
        if isinstance(data, dict):
            data = UserDict(data)
        client = default_client()
        return client.scatter(data, **kwargs)

    @classmethod
    def wait(cls, obj_ids, num_returns=None):
        """
        Wait on the objects without materializing them (blocking operation).

        Parameters
        ----------
        obj_ids : list, scalar
        num_returns : int, optional
        """
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        if num_returns is None:
            num_returns = len(obj_ids)
        if num_returns == len(obj_ids):
            wait(obj_ids, return_when='ALL_COMPLETED')
        else:
            done, not_done = wait(obj_ids, return_when='FIRST_COMPLETED')
            while len(done) < num_returns and (i := (0 < num_returns)):
                extra_done, not_done = wait(not_done, return_when='FIRST_COMPLETED')
                done.update(extra_done)
                i += 1