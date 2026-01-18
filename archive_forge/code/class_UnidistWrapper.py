import asyncio
import unidist
class UnidistWrapper:
    """Mixin that provides means of running functions remotely and getting local results."""

    @classmethod
    def deploy(cls, func, f_args=None, f_kwargs=None, num_returns=1):
        """
        Run local `func` remotely.

        Parameters
        ----------
        func : callable or unidist.ObjectRef
            The function to perform.
        f_args : list or tuple, optional
            Positional arguments to pass to ``func``.
        f_kwargs : dict, optional
            Keyword arguments to pass to ``func``.
        num_returns : int, default: 1
            Amount of return values expected from `func`.

        Returns
        -------
        unidist.ObjectRef or list
            Unidist identifier of the result being put to object store.
        """
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        return _deploy_unidist_func.options(num_returns=num_returns).remote(func, *args, **kwargs)

    @classmethod
    def is_future(cls, item):
        """
        Check if the item is a Future.

        Parameters
        ----------
        item : unidist.ObjectRef or object
            Future or object to check.

        Returns
        -------
        boolean
            If the value is a future.
        """
        return unidist.is_object_ref(item)

    @classmethod
    def materialize(cls, obj_id):
        """
        Get the value of object from the object store.

        Parameters
        ----------
        obj_id : unidist.ObjectRef
            Unidist object identifier to get the value by.

        Returns
        -------
        object
            Whatever was identified by `obj_id`.
        """
        return unidist.get(obj_id)

    @classmethod
    def put(cls, data, **kwargs):
        """
        Put data into the object store.

        Parameters
        ----------
        data : object
            Data to be put.
        **kwargs : dict
            Additional keyword arguments (mostly for compatibility).

        Returns
        -------
        unidist.ObjectRef
            A reference to `data`.
        """
        return unidist.put(data)

    @classmethod
    def wait(cls, obj_ids, num_returns=None):
        """
        Wait on the objects without materializing them (blocking operation).

        ``unidist.wait`` assumes a list of unique object references: see
        https://github.com/modin-project/modin/issues/5045

        Parameters
        ----------
        obj_ids : list, scalar
        num_returns : int, optional
        """
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        unique_ids = list(set(obj_ids))
        if num_returns is None:
            num_returns = len(unique_ids)
        unidist.wait(unique_ids, num_returns=num_returns)