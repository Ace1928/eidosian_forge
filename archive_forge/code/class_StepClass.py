import abc
import json
import logging
import os
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional
import yaml
from mlflow.recipes.cards import CARD_HTML_NAME, CARD_PICKLE_NAME, BaseCard, FailureCard
from mlflow.recipes.utils import get_recipe_name
from mlflow.recipes.utils.step import display_html
from mlflow.tracking import MlflowClient
from mlflow.utils.databricks_utils import is_in_databricks_runtime
class StepClass(Enum):
    """
    Represents the class of a step.
    """
    UNKNOWN = 'UNKNOWN'
    TRAINING = 'TRAINING'
    PREDICTION = 'PREDICTION'