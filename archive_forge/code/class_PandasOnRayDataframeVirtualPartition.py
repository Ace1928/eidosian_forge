import pandas
import ray
from ray.util import get_node_ip_address
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.core.execution.ray.common import RayWrapper
from modin.utils import _inherit_docstrings
from .partition import PandasOnRayDataframePartition
class PandasOnRayDataframeVirtualPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_partitions : Union[list, PandasOnRayDataframePartition]
        List of ``PandasOnRayDataframePartition`` and
        ``PandasOnRayDataframeVirtualPartition`` objects, or a single
        ``PandasOnRayDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    length : ray.ObjectRef or int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : ray.ObjectRef or int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    """
    _PARTITIONS_METADATA_LEN = 3
    partition_type = PandasOnRayDataframePartition
    instance_type = ray.ObjectRef
    axis = None
    _DEPLOY_AXIS_FUNC = None
    _DEPLOY_SPLIT_FUNC = None
    _DRAIN_FUNC = None

    @classmethod
    def _get_deploy_axis_func(cls):
        if cls._DEPLOY_AXIS_FUNC is None:
            cls._DEPLOY_AXIS_FUNC = RayWrapper.put(PandasDataframeAxisPartition.deploy_axis_func)
        return cls._DEPLOY_AXIS_FUNC

    @classmethod
    def _get_deploy_split_func(cls):
        if cls._DEPLOY_SPLIT_FUNC is None:
            cls._DEPLOY_SPLIT_FUNC = RayWrapper.put(PandasDataframeAxisPartition.deploy_splitting_func)
        return cls._DEPLOY_SPLIT_FUNC

    @classmethod
    def _get_drain_func(cls):
        if cls._DRAIN_FUNC is None:
            cls._DRAIN_FUNC = RayWrapper.put(PandasDataframeAxisPartition.drain)
        return cls._DRAIN_FUNC

    @property
    def list_of_ips(self):
        """
        Get the IPs holding the physical objects composing this partition.

        Returns
        -------
        List
            A list of IPs as ``ray.ObjectRef`` or str.
        """
        result = [None] * len(self.list_of_block_partitions)
        for idx, partition in enumerate(self.list_of_block_partitions):
            partition.drain_call_queue()
            result[idx] = partition.ip(materialize=False)
        return result

    @classmethod
    @_inherit_docstrings(PandasDataframeAxisPartition.deploy_splitting_func)
    def deploy_splitting_func(cls, axis, func, f_args, f_kwargs, num_splits, *partitions, extract_metadata=False):
        return _deploy_ray_func.options(num_returns=num_splits * (1 + cls._PARTITIONS_METADATA_LEN) if extract_metadata else num_splits).remote(cls._get_deploy_split_func(), *f_args, num_splits, *partitions, axis=axis, f_to_deploy=func, f_len_args=len(f_args), f_kwargs=f_kwargs, extract_metadata=extract_metadata)

    @classmethod
    def deploy_axis_func(cls, axis, func, f_args, f_kwargs, num_splits, maintain_partitioning, *partitions, min_block_size, lengths=None, manual_partition=False, max_retries=None):
        """
        Deploy a function along a full axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to ``func``.
        f_kwargs : dict
            Keyword arguments to pass to ``func``.
        num_splits : int
            The number of splits to return (see ``split_result_of_axis_func_pandas``).
        maintain_partitioning : bool
            If True, keep the old partitioning if possible.
            If False, create a new partition layout.
        *partitions : iterable
            All partitions that make up the full axis (row or column).
        min_block_size : int
            Minimum number of rows/columns in a single split.
        lengths : list, optional
            The list of lengths to shuffle the object.
        manual_partition : bool, default: False
            If True, partition the result with `lengths`.
        max_retries : int, default: None
            The max number of times to retry the func.

        Returns
        -------
        list
            A list of ``ray.ObjectRef``-s.
        """
        return _deploy_ray_func.options(num_returns=(num_splits if lengths is None else len(lengths)) * (1 + cls._PARTITIONS_METADATA_LEN), **{'max_retries': max_retries} if max_retries is not None else {}).remote(cls._get_deploy_axis_func(), *f_args, num_splits, maintain_partitioning, *partitions, axis=axis, f_to_deploy=func, f_len_args=len(f_args), f_kwargs=f_kwargs, manual_partition=manual_partition, min_block_size=min_block_size, lengths=lengths, return_generator=True)

    @classmethod
    def deploy_func_between_two_axis_partitions(cls, axis, func, f_args, f_kwargs, num_splits, len_of_left, other_shape, *partitions, min_block_size):
        """
        Deploy a function along a full axis between two data sets.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to ``func``.
        f_kwargs : dict
            Keyword arguments to pass to ``func``.
        num_splits : int
            The number of splits to return (see ``split_result_of_axis_func_pandas``).
        len_of_left : int
            The number of values in `partitions` that belong to the left data set.
        other_shape : np.ndarray
            The shape of right frame in terms of partitions, i.e.
            (other_shape[i-1], other_shape[i]) will indicate slice to restore i-1 axis partition.
        *partitions : iterable
            All partitions that make up the full axis (row or column) for both data sets.
        min_block_size : int
            Minimum number of rows/columns in a single split.

        Returns
        -------
        list
            A list of ``ray.ObjectRef``-s.
        """
        return _deploy_ray_func.options(num_returns=num_splits * (1 + cls._PARTITIONS_METADATA_LEN)).remote(PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions, *f_args, num_splits, len_of_left, other_shape, *partitions, axis=axis, f_to_deploy=func, f_len_args=len(f_args), f_kwargs=f_kwargs, min_block_size=min_block_size, return_generator=True)

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        futures = self.list_of_blocks
        RayWrapper.wait(futures)