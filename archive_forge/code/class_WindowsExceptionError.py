import io
import pickle
from typing import Union, Any
from pydantic import BaseModel
from pydantic.types import ByteSize
class WindowsExceptionError(Exception):
    """Windows error place-holder on platforms without support."""