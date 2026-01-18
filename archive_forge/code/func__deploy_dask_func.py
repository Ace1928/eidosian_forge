import pandas
from distributed import Future
from distributed.utils import get_ip
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.core.execution.dask.common import DaskWrapper
from modin.utils import _inherit_docstrings
from .partition import PandasOnDaskDataframePartition
def _deploy_dask_func(deployer, axis, f_to_deploy, f_args, f_kwargs, *args, extract_metadata=True, **kwargs):
    """
    Execute a function on an axis partition in a worker process.

    This is ALWAYS called on either ``PandasDataframeAxisPartition.deploy_axis_func``
    or ``PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions``, which both
    serve to deploy another dataframe function on a Dask worker process.

    Parameters
    ----------
    deployer : callable
        A `PandasDataFrameAxisPartition.deploy_*` method that will call `deploy_f`.
    axis : {0, 1}
        The axis to perform the function along.
    f_to_deploy : callable or RayObjectID
        The function to deploy.
    f_args : list or tuple
        Positional arguments to pass to ``f_to_deploy``.
    f_kwargs : dict
        Keyword arguments to pass to ``f_to_deploy``.
    *args : list
        Positional arguments to pass to ``func``.
    extract_metadata : bool, default: True
        Whether to return metadata (length, width, ip) of the result. Passing `False` may relax
        the load on object storage as the remote function would return 4 times fewer futures.
        Passing `False` makes sense for temporary results where you know for sure that the
        metadata will never be requested.
    **kwargs : dict
        Keyword arguments to pass to ``func``.

    Returns
    -------
    list
        The result of the function ``func`` and metadata for it.
    """
    result = deployer(axis, f_to_deploy, f_args, f_kwargs, *args, **kwargs)
    if not extract_metadata:
        return result
    ip = get_ip()
    if isinstance(result, pandas.DataFrame):
        return (result, len(result), len(result.columns), ip)
    elif all((isinstance(r, pandas.DataFrame) for r in result)):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]