from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing_extensions import TYPE_CHECKING, Literal
from langchain_community.vectorstores.redis.constants import REDIS_VECTOR_DTYPE_MAP
def _fields(self) -> Dict[str, Any]:
    field_data = {'TYPE': self.datatype, 'DIM': self.dims, 'DISTANCE_METRIC': self.distance_metric}
    if self.initial_cap is not None:
        field_data['INITIAL_CAP'] = self.initial_cap
    return field_data