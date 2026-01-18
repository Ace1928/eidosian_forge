import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class ReportViewspec(ReportAPIBaseModel):
    id: str = ''
    name: str = ''
    display_name: str = ''
    description: str = ''
    project: Project = Field(default_factory=Project)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    spec: Spec = Field(default_factory=Spec)

    @validator('spec', pre=True)
    def parse_json(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError('invalid json')
        return v