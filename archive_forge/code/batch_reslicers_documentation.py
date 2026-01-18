import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
Reslice the batch stream into new batches, each containing the same keys

        :param batches: the batch stream

        :yield: an iterable of batches, each containing the same keys
        