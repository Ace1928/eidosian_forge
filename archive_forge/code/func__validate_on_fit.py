from typing import Dict, Iterable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import Max, Min
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def _validate_on_fit(self):
    self._validate_bins_columns()