import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class MediaBrowserConfig(ReportAPIBaseModel):
    column_count: Optional[int] = None
    media_keys: LList[str] = Field(default_factory=list)