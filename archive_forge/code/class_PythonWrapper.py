class PythonWrapper:
    """Python engine wrapper serving for the compatibility purpose with other engines."""

    @classmethod
    def deploy(cls, func, f_args=None, f_kwargs=None, num_returns=1):
        """
        Run the passed function.

        Parameters
        ----------
        func : callable
        f_args : sequence, optional
            Positional arguments to pass to the `func`.
        f_kwargs : dict, optional
            Keyword arguments to pass to the `func`.
        num_returns : int, default: 1
            Number of return values from the `func`.

        Returns
        -------
        object
            Returns the result of the `func`.
        """
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        return func(*args, **kwargs)

    @classmethod
    def is_future(cls, item):
        """
        Check if the item is a Future.

        Parameters
        ----------
        item : object

        Returns
        -------
        boolean
            Always return false.
        """
        return False

    @classmethod
    def materialize(cls, obj_id):
        """
        Get the data from the data storage.

        The method only serves for the compatibility purpose, what it actually
        does is just return the passed value as is.

        Parameters
        ----------
        obj_id : object

        Returns
        -------
        object
            The passed `obj_id` itself.
        """
        return obj_id

    @classmethod
    def put(cls, data, **kwargs):
        """
        Put data into the data storage.

        The method only serves for the compatibility purpose, what it actually
        does is just return the passed value as is.

        Parameters
        ----------
        data : object
        **kwargs : dict

        Returns
        -------
        object
            The passed `data` itself.
        """
        return data