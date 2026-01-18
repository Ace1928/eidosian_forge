import warnings
import pandas
import unidist
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.unidist.common import UnidistWrapper
from modin.core.execution.unidist.common.utils import deserialize
from modin.logging import get_logger
from modin.pandas.indexing import compute_sliced_len
class PandasOnUnidistDataframePartition(PandasDataframePartition):
    """
    The class implements the interface in ``PandasDataframePartition``.

    Parameters
    ----------
    data : unidist.ObjectRef
        A reference to ``pandas.DataFrame`` that need to be wrapped with this class.
    length : unidist.ObjectRef or int, optional
        Length or reference to it of wrapped ``pandas.DataFrame``.
    width : unidist.ObjectRef or int, optional
        Width or reference to it of wrapped ``pandas.DataFrame``.
    ip : unidist.ObjectRef or str, optional
        Node IP address or reference to it that holds wrapped ``pandas.DataFrame``.
    call_queue : list
        Call queue that needs to be executed on wrapped ``pandas.DataFrame``.
    """
    execution_wrapper = UnidistWrapper

    def __init__(self, data, length=None, width=None, ip=None, call_queue=None):
        super().__init__()
        assert unidist.is_object_ref(data)
        self._data = data
        self.call_queue = call_queue if call_queue is not None else []
        self._length_cache = length
        self._width_cache = width
        self._ip_cache = ip
        log = get_logger()
        self._is_debug(log) and log.debug('Partition ID: {}, Height: {}, Width: {}, Node IP: {}'.format(self._identity, str(self._length_cache), str(self._width_cache), str(self._ip_cache)))

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to the object wrapped by this partition.

        Parameters
        ----------
        func : callable or unidist.ObjectRef
            A function to apply.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnUnidistDataframePartition
            A new ``PandasOnUnidistDataframePartition`` object.

        Notes
        -----
        It does not matter if `func` is callable or an ``unidist.ObjectRef``. Unidist will
        handle it correctly either way. The keyword arguments are sent as a dictionary.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f'ENTER::Partition.apply::{self._identity}')
        data = self._data
        call_queue = self.call_queue + [[func, args, kwargs]]
        if len(call_queue) > 1:
            self._is_debug(log) and log.debug(f'SUBMIT::_apply_list_of_funcs::{self._identity}')
            result, length, width, ip = _apply_list_of_funcs.remote(call_queue, data)
        else:
            result, length, width, ip = _apply_func.remote(data, func, *args, **kwargs)
            self._is_debug(log) and log.debug(f'SUBMIT::_apply_func::{self._identity}')
        self._is_debug(log) and log.debug(f'EXIT::Partition.apply::{self._identity}')
        return self.__constructor__(result, length, width, ip)

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        log = get_logger()
        self._is_debug(log) and log.debug(f'ENTER::Partition.drain_call_queue::{self._identity}')
        if len(self.call_queue) == 0:
            return
        data = self._data
        call_queue = self.call_queue
        if len(call_queue) > 1:
            self._is_debug(log) and log.debug(f'SUBMIT::_apply_list_of_funcs::{self._identity}')
            self._data, new_length, new_width, self._ip_cache = _apply_list_of_funcs.remote(call_queue, data)
        else:
            func, f_args, f_kwargs = call_queue[0]
            self._is_debug(log) and log.debug(f'SUBMIT::_apply_func::{self._identity}')
            self._data, new_length, new_width, self._ip_cache = _apply_func.remote(data, func, *f_args, **f_kwargs)
        self._is_debug(log) and log.debug(f'EXIT::Partition.drain_call_queue::{self._identity}')
        self.call_queue = []
        if not isinstance(self._length_cache, int):
            self._length_cache = new_length
        if not isinstance(self._width_cache, int):
            self._width_cache = new_width

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        UnidistWrapper.wait(self._data)

    def mask(self, row_labels, col_labels):
        """
        Lazily create a mask that extracts the indices provided.

        Parameters
        ----------
        row_labels : list-like, slice or label
            The row labels for the rows to extract.
        col_labels : list-like, slice or label
            The column labels for the columns to extract.

        Returns
        -------
        PandasOnUnidistDataframePartition
            A new ``PandasOnUnidistDataframePartition`` object.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f'ENTER::Partition.mask::{self._identity}')
        new_obj = super().mask(row_labels, col_labels)
        if isinstance(row_labels, slice) and unidist.is_object_ref(self._length_cache):
            if row_labels == slice(None):
                new_obj._length_cache = self._length_cache
            else:
                new_obj._length_cache = compute_sliced_len.remote(row_labels, self._length_cache)
        if isinstance(col_labels, slice) and unidist.is_object_ref(self._width_cache):
            if col_labels == slice(None):
                new_obj._width_cache = self._width_cache
            else:
                new_obj._width_cache = compute_sliced_len.remote(col_labels, self._width_cache)
        self._is_debug(log) and log.debug(f'EXIT::Partition.mask::{self._identity}')
        return new_obj

    @classmethod
    def put(cls, obj):
        """
        Put an object into object store and wrap it with partition object.

        Parameters
        ----------
        obj : any
            An object to be put.

        Returns
        -------
        PandasOnUnidistDataframePartition
            A new ``PandasOnUnidistDataframePartition`` object.
        """
        return cls(cls.execution_wrapper.put(obj), len(obj.index), len(obj.columns))

    @classmethod
    def preprocess_func(cls, func):
        """
        Put a function into the object store to use in ``apply``.

        Parameters
        ----------
        func : callable
            A function to preprocess.

        Returns
        -------
        unidist.ObjectRef
            A reference to `func`.
        """
        return cls.execution_wrapper.put(func)

    def length(self, materialize=True):
        """
        Get the length of the object wrapped by this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        int or unidist.ObjectRef
            The length of the object.
        """
        if self._length_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = _get_index_and_columns_size.remote(self._data)
        if unidist.is_object_ref(self._length_cache) and materialize:
            self._length_cache = UnidistWrapper.materialize(self._length_cache)
        return self._length_cache

    def width(self, materialize=True):
        """
        Get the width of the object wrapped by the partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        int or unidist.ObjectRef
            The width of the object.
        """
        if self._width_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = _get_index_and_columns_size.remote(self._data)
        if unidist.is_object_ref(self._width_cache) and materialize:
            self._width_cache = UnidistWrapper.materialize(self._width_cache)
        return self._width_cache

    def ip(self, materialize=True):
        """
        Get the node IP address of the object wrapped by this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        str
            IP address of the node that holds the data.
        """
        if self._ip_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._ip_cache = self.apply(lambda df: pandas.DataFrame([]))._ip_cache
        if materialize and unidist.is_object_ref(self._ip_cache):
            self._ip_cache = UnidistWrapper.materialize(self._ip_cache)
        return self._ip_cache