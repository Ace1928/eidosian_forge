import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class ScalarChartConfig(ReportAPIBaseModel):
    chart_title: Optional[str] = None
    metrics: LList[str] = Field(default_factory=list)
    group_agg: Optional[GroupAgg] = None
    group_area: Optional[GroupArea] = None
    expressions: Optional[LList[str]] = None
    legend_template: Optional[str] = None
    font_size: Optional[FontSize] = None