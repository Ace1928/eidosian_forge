from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from modin.logging import ClassLogger
def _wrap_partitions(self, partitions: list, extract_metadata: Optional[bool]=None) -> list:
    """
        Wrap remote partition objects with `BaseDataframePartition` class.

        Parameters
        ----------
        partitions : list
            List of remotes partition objects to be wrapped with `BaseDataframePartition` class.
        extract_metadata : bool, optional
            Whether the partitions list contains information about partition's metadata.
            If `None` was passed will take the argument's value from the value of `cls._PARTITIONS_METADATA_LEN`.

        Returns
        -------
        list
            List of wrapped remote partition objects.
        """
    assert self.partition_type is not None
    assert self.instance_type is not None
    if extract_metadata is None:
        extract_metadata = bool(self._PARTITIONS_METADATA_LEN)
    if extract_metadata:
        return [self.partition_type(*init_args) for init_args in zip(*[iter(partitions)] * (1 + self._PARTITIONS_METADATA_LEN))]
    else:
        return [self.partition_type(object_id) for object_id in partitions]