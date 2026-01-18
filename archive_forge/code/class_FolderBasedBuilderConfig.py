import collections
import itertools
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type
import pandas as pd
import pyarrow as pa
import pyarrow.json as paj
import datasets
from datasets.features.features import FeatureType
from datasets.tasks.base import TaskTemplate
@dataclass
class FolderBasedBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for AutoFolder."""
    features: Optional[datasets.Features] = None
    drop_labels: bool = None
    drop_metadata: bool = None