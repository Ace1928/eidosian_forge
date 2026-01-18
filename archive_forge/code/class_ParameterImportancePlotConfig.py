import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class ParameterImportancePlotConfig(ReportAPIBaseModel):
    target_key: str
    columns_pinned: dict = Field(default_factory=dict)
    column_widths: dict = Field(default_factory=dict)
    column_order: LList[str] = Field(default_factory=list)
    page_size: int = 10
    only_show_selected: bool = False