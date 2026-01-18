import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class FlowConfig(ReportAPIBaseModel):
    snap_to_columns: bool = True
    columns_per_page: int = 3
    rows_per_page: int = 2
    gutter_width: int = 16
    box_width: int = 460
    box_height: int = 300