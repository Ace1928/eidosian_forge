import re
import datetime
import numpy as np
import csv
import ctypes
@staticmethod
def _get_nom_val(atrv):
    """Given a string containing a nominal type, returns a tuple of the
        possible values.

        A nominal type is defined as something framed between braces ({}).

        Parameters
        ----------
        atrv : str
           Nominal type definition

        Returns
        -------
        poss_vals : tuple
           possible values

        Examples
        --------
        >>> from scipy.io.arff._arffread import NominalAttribute
        >>> NominalAttribute._get_nom_val("{floup, bouga, fl, ratata}")
        ('floup', 'bouga', 'fl', 'ratata')
        """
    m = r_nominal.match(atrv)
    if m:
        attrs, _ = split_data_line(m.group(1))
        return tuple(attrs)
    else:
        raise ValueError('This does not look like a nominal string')