import abc
import uuid
from typing import List, Tuple
import numpy as np
import pyarrow as pa
from modin.error_message import ErrorMessage
class DbTable(abc.ABC):
    """
    Base class, representing a table in the HDK database.

    Attributes
    ----------
    name : str
        Table name.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        """
        Return a tuple with the number of rows and columns.

        Returns
        -------
        tuple of int
        """
        pass

    @property
    @abc.abstractmethod
    def column_names(self) -> List[str]:
        """
        Return a list of the table column names.

        Returns
        -------
        tuple of str
        """
        pass

    @abc.abstractmethod
    def to_arrow(self) -> pa.Table:
        """
        Convert this table to arrow.

        Returns
        -------
        pyarrow.Table
        """
        pass

    def __len__(self):
        """
        Return the number of rows in the table.

        Returns
        -------
        int
        """
        return self.shape[0]