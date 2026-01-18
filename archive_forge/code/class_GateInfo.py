from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
class GateInfo(Operator):
    parameters: List[Union[float, str]] = Field(default_factory=list)
    arguments: List[Union[int, str]] = Field(default_factory=list)
    operator_type: Literal['gate'] = 'gate'