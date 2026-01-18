from datetime import date, datetime
from typing import Any
from unittest import TestCase
import copy
import numpy as np
import pandas as pd
from fugue.bag import Bag, LocalBag
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
from pytest import raises
from triad.collections.schema import Schema
DataFrame level general test suite.
    All new DataFrame types should pass this test suite.
    