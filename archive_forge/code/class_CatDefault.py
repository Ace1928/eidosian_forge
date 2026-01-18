from .series import SeriesDefault
class CatDefault(SeriesDefault):
    """Builder for default-to-pandas methods which is executed under category accessor."""

    @classmethod
    def frame_wrapper(cls, df):
        """
        Get category accessor of the passed frame.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.core.arrays.categorical.CategoricalAccessor
        """
        return df.squeeze(axis=1).cat