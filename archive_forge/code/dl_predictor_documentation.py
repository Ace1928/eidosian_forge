import abc
from typing import Dict, Optional, TypeVar, Union
import numpy as np
import pandas as pd
from ray.air.util.data_batch_conversion import (
from ray.train.predictor import Predictor
from ray.util.annotations import DeveloperAPI
Inputs the tensor to the model for this Predictor and returns the result.

        Args:
            inputs: The tensor to input to the model.

        Returns:
            A tensor or dictionary of tensors containing the model output.
        