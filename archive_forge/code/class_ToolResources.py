from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
class ToolResources(BaseModel):
    code_interpreter: Optional[ToolResourcesCodeInterpreter] = None
    file_search: Optional[ToolResourcesFileSearch] = None