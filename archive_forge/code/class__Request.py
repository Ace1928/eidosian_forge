from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
class _Request(BaseModel):
    jsonrpc: StrictStr = Field('2.0', const=True, example='2.0')
    id: Union[StrictStr, int] = Field(None, example=0)
    method: StrictStr = Field(name, const=True, example=name)

    class Config:
        extra = 'forbid'