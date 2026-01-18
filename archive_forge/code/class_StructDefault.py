from .series import SeriesDefault
class StructDefault(SeriesDefault):
    """Builder for default-to-pandas methods which is executed under struct accessor."""

    @classmethod
    def frame_wrapper(cls, df):
        """
        Get struct accessor of the passed frame.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.core.arrays.arrow.StructAccessor
        """
        return df.squeeze(axis=1).struct