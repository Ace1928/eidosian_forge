import abc
import uuid
from typing import List, Tuple
import numpy as np
import pyarrow as pa
from modin.error_message import ErrorMessage

        Import ``pandas.DataFrame`` to the worker.

        Parameters
        ----------
        df : pandas.DataFrame
            A frame to import.
        name : str, optional
            A table name to use. None to generate a unique name.

        Returns
        -------
        DbTable
            Imported table.
        