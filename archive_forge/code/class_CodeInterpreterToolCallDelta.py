from typing import List, Union, Optional
from typing_extensions import Literal, Annotated
from ....._utils import PropertyInfo
from ....._models import BaseModel
from .code_interpreter_logs import CodeInterpreterLogs
from .code_interpreter_output_image import CodeInterpreterOutputImage
class CodeInterpreterToolCallDelta(BaseModel):
    index: int
    'The index of the tool call in the tool calls array.'
    type: Literal['code_interpreter']
    'The type of tool call.\n\n    This is always going to be `code_interpreter` for this type of tool call.\n    '
    id: Optional[str] = None
    'The ID of the tool call.'
    code_interpreter: Optional[CodeInterpreter] = None
    'The Code Interpreter tool call definition.'