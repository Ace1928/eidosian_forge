import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class LocalPanelSettings(ReportAPIBaseModel):
    x_axis: str = '_step'
    smoothing_weight: int = 0
    smoothing_type: str = 'exponential'
    ignore_outliers: bool = False
    x_axis_active: bool = False
    smoothing_active: bool = False
    ref: Optional[Ref] = None