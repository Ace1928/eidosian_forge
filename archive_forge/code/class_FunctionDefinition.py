from typing import Optional
from ..._models import BaseModel
from .function_parameters import FunctionParameters
class FunctionDefinition(BaseModel):
    name: str
    'The name of the function to be called.\n\n    Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length\n    of 64.\n    '
    description: Optional[str] = None
    '\n    A description of what the function does, used by the model to choose when and\n    how to call the function.\n    '
    parameters: Optional[FunctionParameters] = None
    'The parameters the functions accepts, described as a JSON Schema object.\n\n    See the\n    [guide](https://platform.openai.com/docs/guides/text-generation/function-calling)\n    for examples, and the\n    [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for\n    documentation about the format.\n\n    Omitting `parameters` defines a function with an empty parameter list.\n    '