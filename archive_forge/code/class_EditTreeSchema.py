from collections import defaultdict
from typing import Any, Dict, List, Union
class EditTreeSchema(BaseModel):
    __root__: Union[MatchNodeSchema, SubstNodeSchema]