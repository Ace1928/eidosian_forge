import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class RunFeed(ReportAPIBaseModel):
    version: int = 2
    column_visible: Dict[str, bool] = Field(default_factory=lambda: {'run:name': False})
    column_pinned: Dict[str, bool] = Field(default_factory=dict)
    column_widths: Dict[str, int] = Field(default_factory=dict)
    column_order: LList[str] = Field(default_factory=list)
    page_size: int = 10
    only_show_selected: bool = False