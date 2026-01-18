from .default import DefaultMethod
class ExpandingDefault(DefaultMethod):
    """Builder for default-to-pandas aggregation on an expanding window functions."""
    OBJECT_TYPE = 'Expanding'

    @classmethod
    def _build_expanding(cls, func, squeeze_self):
        """
        Build function that creates an expanding window and executes `func` on it.

        Parameters
        ----------
        func : callable
            Function to execute on a expanding window.
        squeeze_self : bool
            Whether or not to squeeze frame before executing the window function.

        Returns
        -------
        callable
            Function that takes pandas DataFrame and applies `func` on a expanding window.
        """

        def fn(df, rolling_args, *args, **kwargs):
            """Create rolling window for the passed frame and execute specified `func` on it."""
            if squeeze_self:
                df = df.squeeze(axis=1)
            roller = df.expanding(*rolling_args)
            if type(func) is property:
                return func.fget(roller)
            return func(roller, *args, **kwargs)
        return fn

    @classmethod
    def register(cls, func, squeeze_self=False, **kwargs):
        """
        Build function that do fallback to pandas to apply `func` on a expanding window.

        Parameters
        ----------
        func : callable
            Function to execute on an expanding window.
        squeeze_self : bool, default: False
            Whether or not to squeeze frame before executing the window function.
        **kwargs : kwargs
            Additional arguments that will be passed to function builder.

        Returns
        -------
        callable
            Function that takes query compiler and defaults to pandas to apply aggregation
            `func` on an expanding window.
        """
        return super().register(cls._build_expanding(func, squeeze_self=squeeze_self), fn_name=func.__name__, **kwargs)