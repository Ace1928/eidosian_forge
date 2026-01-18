import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class UserQuery(ReportAPIBaseModel):
    query_fields: LList[QueryField] = Field(default_factory=lambda: [QueryField()])