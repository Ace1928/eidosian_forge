from typing import List, Union
from typing_extensions import Literal, Annotated
from ....._utils import PropertyInfo
from ....._models import BaseModel
class CodeInterpreterToolCall(BaseModel):
    id: str
    'The ID of the tool call.'
    code_interpreter: CodeInterpreter
    'The Code Interpreter tool call definition.'
    type: Literal['code_interpreter']
    'The type of tool call.\n\n    This is always going to be `code_interpreter` for this type of tool call.\n    '