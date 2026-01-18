import warnings
import numpy as np
import pandas as pd
from scipy import stats
def _do_ros(df, observations, censorship, transform_in, transform_out):
    """
    DataFrame-centric function to impute censored valies with ROS.

    Prepares a dataframe for, and then esimates the values of a censored
    dataset using Regression on Order Statistics

    Parameters
    ----------
    df : DataFrame

    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    transform_in, transform_out : callable
        Transformations to be applied to the data prior to fitting
        the line and after estimated values from that line. Typically,
        `np.log` and `np.exp` are used, respectively.

    Returns
    -------
    estimated : DataFrame
        A new dataframe with two new columns: "estimated" and "final".
        The "estimated" column contains of the values inferred from the
        best-fit line. The "final" column contains the estimated values
        only where the original observations were censored, and the original
        observations everwhere else.
    """
    cohn = cohn_numbers(df, observations=observations, censorship=censorship)
    modeled = _ros_sort(df, observations=observations, censorship=censorship)
    modeled.loc[:, 'det_limit_index'] = modeled[observations].apply(_detection_limit_index, args=(cohn,))
    modeled.loc[:, 'rank'] = _ros_group_rank(modeled, 'det_limit_index', censorship)
    modeled.loc[:, 'plot_pos'] = plotting_positions(modeled, censorship, cohn)
    modeled.loc[:, 'Zprelim'] = stats.norm.ppf(modeled['plot_pos'])
    return _impute(modeled, observations, censorship, transform_in, transform_out)