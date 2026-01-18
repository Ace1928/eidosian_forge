import time
def _add_new_columns(dataframe, metrics):
    """Add new metrics as new columns to selected pandas dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Selected dataframe needs to be modified.
    metrics : metric.EvalMetric
        New metrics to be added.
    """
    new_columns = set(metrics.keys()) - set(dataframe.columns)
    for col in new_columns:
        dataframe[col] = None