from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
class CompilerISA(BaseModel):
    qubits: Dict[str, Qubit] = Field(default_factory=dict, alias='1Q')
    edges: Dict[str, Edge] = Field(default_factory=dict, alias='2Q')