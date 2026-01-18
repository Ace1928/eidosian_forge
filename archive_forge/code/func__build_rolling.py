from .default import DefaultMethod
@classmethod
def _build_rolling(cls, func):
    """
        Build function that creates a rolling window and executes `func` on it.

        Parameters
        ----------
        func : callable
            Function to execute on a rolling window.

        Returns
        -------
        callable
            Function that takes pandas DataFrame and applies `func` on a rolling window.
        """

    def fn(df, rolling_kwargs, *args, **kwargs):
        """Create rolling window for the passed frame and execute specified `func` on it."""
        roller = df.rolling(**rolling_kwargs)
        if type(func) is property:
            return func.fget(roller)
        return func(roller, *args, **kwargs)
    return fn