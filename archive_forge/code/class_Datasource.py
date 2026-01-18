import warnings
from typing import Any, Callable, Iterable, List, Optional
import numpy as np
import ray
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@PublicAPI
class Datasource:
    """Interface for defining a custom :class:`~ray.data.Dataset` datasource.

    To read a datasource into a dataset, use :meth:`~ray.data.read_datasource`.
    """

    @Deprecated
    def create_reader(self, **read_args) -> 'Reader':
        """Return a Reader for the given read arguments.

        The reader object will be responsible for querying the read metadata, and
        generating the actual read tasks to retrieve the data blocks upon request.

        Args:
            read_args: Additional kwargs to pass to the datasource impl.
        """
        warnings.warn('`create_reader` has been deprecated in Ray 2.9. Instead of creating a `Reader`, implement `Datasource.get_read_tasks` and `Datasource.estimate_inmemory_data_size`.', DeprecationWarning)
        return _LegacyDatasourceReader(self, **read_args)

    @Deprecated
    def prepare_read(self, parallelism: int, **read_args) -> List['ReadTask']:
        """Deprecated: Please implement create_reader() instead."""
        raise NotImplementedError

    @Deprecated
    def on_write_start(self, **write_args) -> None:
        """Callback for when a write job starts.

        Use this method to perform setup for write tasks. For example, creating a
        staging bucket in S3.

        Args:
            write_args: Additional kwargs to pass to the datasource impl.
        """
        pass

    @Deprecated
    def write(self, blocks: Iterable[Block], ctx: TaskContext, **write_args) -> WriteResult:
        """Write blocks out to the datasource. This is used by a single write task.

        Args:
            blocks: List of data blocks.
            ctx: ``TaskContext`` for the write task.
            write_args: Additional kwargs to pass to the datasource impl.

        Returns:
            The output of the write task.
        """
        raise NotImplementedError

    @Deprecated
    def on_write_complete(self, write_results: List[WriteResult], **kwargs) -> None:
        """Callback for when a write job completes.

        This can be used to "commit" a write output. This method must
        succeed prior to ``write_datasource()`` returning to the user. If this
        method fails, then ``on_write_failed()`` will be called.

        Args:
            write_results: The list of the write task results.
            kwargs: Forward-compatibility placeholder.
        """
        pass

    @Deprecated
    def on_write_failed(self, write_results: List[ObjectRef[WriteResult]], error: Exception, **kwargs) -> None:
        """Callback for when a write job fails.

        This is called on a best-effort basis on write failures.

        Args:
            write_results: The list of the write task result futures.
            error: The first error encountered.
            kwargs: Forward-compatibility placeholder.
        """
        pass

    def get_name(self) -> str:
        """Return a human-readable name for this datasource.
        This will be used as the names of the read tasks.
        """
        name = type(self).__name__
        datasource_suffix = 'Datasource'
        if name.endswith(datasource_suffix):
            name = name[:-len(datasource_suffix)]
        return name

    def estimate_inmemory_data_size(self) -> Optional[int]:
        """Return an estimate of the in-memory data size, or None if unknown.

        Note that the in-memory data size may be larger than the on-disk data size.
        """
        raise NotImplementedError

    def get_read_tasks(self, parallelism: int) -> List['ReadTask']:
        """Execute the read and return read tasks.

        Args:
            parallelism: The requested read parallelism. The number of read
                tasks should equal to this value if possible.

        Returns:
            A list of read tasks that can be executed to read blocks from the
            datasource in parallel.
        """
        raise NotImplementedError

    @property
    def should_create_reader(self) -> bool:
        has_implemented_get_read_tasks = type(self).get_read_tasks is not Datasource.get_read_tasks
        has_implemented_estimate_inmemory_data_size = type(self).estimate_inmemory_data_size is not Datasource.estimate_inmemory_data_size
        return not has_implemented_get_read_tasks or not has_implemented_estimate_inmemory_data_size

    @property
    def supports_distributed_reads(self) -> bool:
        """If ``False``, only launch read tasks on the driver's node."""
        return True