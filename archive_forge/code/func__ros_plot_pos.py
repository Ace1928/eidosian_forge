import warnings
import numpy as np
import pandas as pd
from scipy import stats
def _ros_plot_pos(row, censorship, cohn):
    """
    ROS-specific plotting positions.

    Computes the plotting position for an observation based on its rank,
    censorship status, and detection limit index.

    Parameters
    ----------
    row : {Series, dict}
        Full observation (row) from a censored dataset. Requires a
        'rank', 'detection_limit', and `censorship` column.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    cohn : DataFrame
        DataFrame of Cohn numbers.

    Returns
    -------
    plotting_position : float

    See Also
    --------
    cohn_numbers
    """
    DL_index = row['det_limit_index']
    rank = row['rank']
    censored = row[censorship]
    dl_1 = cohn.iloc[DL_index]
    dl_2 = cohn.iloc[DL_index + 1]
    if censored:
        return (1 - dl_1['prob_exceedance']) * rank / (dl_1['ncen_equal'] + 1)
    else:
        return 1 - dl_1['prob_exceedance'] + (dl_1['prob_exceedance'] - dl_2['prob_exceedance']) * rank / (dl_1['nuncen_above'] + 1)