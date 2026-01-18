from typing import List, Union
from typing_extensions import Literal, Annotated
from ....._utils import PropertyInfo
from ....._models import BaseModel
class CodeInterpreter(BaseModel):
    input: str
    'The input to the Code Interpreter tool call.'
    outputs: List[CodeInterpreterOutput]
    'The outputs from the Code Interpreter tool call.\n\n    Code Interpreter can output one or more items, including text (`logs`) or images\n    (`image`). Each of these are represented by a different object type.\n    '