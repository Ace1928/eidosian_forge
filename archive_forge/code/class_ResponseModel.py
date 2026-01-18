from typing import Any, Dict, List
from pydantic import BaseModel
class ResponseModel(BaseModel, extra='ignore'):
    """
    A pydantic model representing Gateway response data, such as information about a Gateway
    Route returned in response to a GetRoute request
    """