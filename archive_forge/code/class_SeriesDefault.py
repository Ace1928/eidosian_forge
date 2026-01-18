from .default import DefaultMethod
class SeriesDefault(DefaultMethod):
    """Builder for default-to-pandas methods which is executed under Series."""
    OBJECT_TYPE = 'Series'

    @classmethod
    def frame_wrapper(cls, df):
        """
        Squeeze passed DataFrame to be able to process Series-specific functions on it.

        Parameters
        ----------
        df : pandas.DataFrame
            One-column DataFrame to squeeze.

        Returns
        -------
        pandas.Series
        """
        return df.squeeze(axis=1)