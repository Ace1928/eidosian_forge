from .series import SeriesDefault
class StrDefault(SeriesDefault):
    """Builder for default-to-pandas methods which is executed under `str` accessor."""

    @classmethod
    def frame_wrapper(cls, df):
        """
        Get `str` accessor of the passed frame.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.core.strings.accessor.StringMethods
        """
        return df.squeeze(axis=1).str