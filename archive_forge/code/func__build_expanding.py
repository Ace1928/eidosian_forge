from .default import DefaultMethod
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