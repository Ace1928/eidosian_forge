from typing import Any, Dict, List
from pydantic import BaseModel
class SetLimitsModel(BaseModel, extra='ignore'):
    route: str
    limits: List[Dict[str, Any]]
    '\n    A pydantic model representing Gateway SetLimits request body, containing route and limits.\n    '