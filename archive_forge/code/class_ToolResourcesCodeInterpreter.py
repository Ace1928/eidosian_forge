from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
class ToolResourcesCodeInterpreter(BaseModel):
    file_ids: Optional[List[str]] = None
    '\n    A list of [file](https://platform.openai.com/docs/api-reference/files) IDs made\n    available to the `code_interpreter` tool. There can be a maximum of 20 files\n    associated with the tool.\n    '