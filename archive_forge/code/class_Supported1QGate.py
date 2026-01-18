from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
class Supported1QGate:
    I = 'I'
    RX = 'RX'
    RZ = 'RZ'
    MEASURE = 'MEASURE'
    WILDCARD = 'WILDCARD'