import warnings
import numpy as np
import pandas as pd
from scipy import stats
def _impute(df, observations, censorship, transform_in, transform_out):
    """
    Executes the basic regression on order stat (ROS) proceedure.

    Uses ROS to impute censored from the best-fit line of a
    probability plot of the uncensored values.

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
    uncensored_mask = ~df[censorship]
    censored_mask = df[censorship]
    fit_params = stats.linregress(df['Zprelim'][uncensored_mask], transform_in(df[observations][uncensored_mask]))
    slope, intercept = fit_params[:2]
    df.loc[:, 'estimated'] = transform_out(slope * df['Zprelim'][censored_mask] + intercept)
    df.loc[:, 'final'] = np.where(df[censorship], df['estimated'], df[observations])
    return df