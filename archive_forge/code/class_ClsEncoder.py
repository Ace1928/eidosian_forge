import json
import logging
from typing import Any
from fastapi import Request
from pydantic import BaseModel
from enum import Enum
class ClsEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.dict()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)