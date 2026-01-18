import datetime
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from unittest import TestCase
from uuid import uuid4
from triad.utils.io import write_text, join
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest import raises
from triad import SerializableRLock
import fugue.api as fa
from fugue import (
from fugue.column import col
from fugue.column import functions as ff
from fugue.column import lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import (
class MockEngine(QPDPandasEngine):

    def __init__(self, execution_engine, p: int=0):
        super().__init__(execution_engine)
        self.p = p

    def select(self, dfs, statement):
        assert 2 == self.p
        return super().select(dfs, statement)