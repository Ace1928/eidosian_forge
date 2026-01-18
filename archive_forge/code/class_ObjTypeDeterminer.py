import pandas
from pandas.core.dtypes.common import is_list_like
from modin.core.dataframe.algebra.operator import Operator
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
class ObjTypeDeterminer:
    """
    Class that routes work to the frame.

    Provides an instance which forwards all of the `__getattribute__` calls
    to an object under which `key` function is applied.
    """

    def __getattr__(self, key):
        """
        Build function that executes `key` function over passed frame.

        Parameters
        ----------
        key : str

        Returns
        -------
        callable
            Function that takes DataFrame and executes `key` function on it.
        """

        def func(df, *args, **kwargs):
            """Access specified attribute of the passed object and call it if it's callable."""
            prop = getattr(df, key)
            if callable(prop):
                return prop(*args, **kwargs)
            else:
                return prop
        return func