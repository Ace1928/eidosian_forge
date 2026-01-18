from .default import DefaultMethod
class ResampleDefault(DefaultMethod):
    """Builder for default-to-pandas resampled aggregation functions."""
    OBJECT_TYPE = 'Resampler'

    @classmethod
    def register(cls, func, squeeze_self=False, **kwargs):
        """
        Build function that do fallback to pandas and aggregate resampled data.

        Parameters
        ----------
        func : callable
            Aggregation function to execute under resampled frame.
        squeeze_self : bool, default: False
            Whether or not to squeeze frame before resampling.
        **kwargs : kwargs
            Additional arguments that will be passed to function builder.

        Returns
        -------
        callable
            Function that takes query compiler and does fallback to pandas to resample
            time-series data and apply aggregation on it.
        """
        return super().register(Resampler.build_resample(func, squeeze_self), fn_name=func.__name__, **kwargs)