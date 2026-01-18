from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
class MyEdge(ObjectType):
    node = Field(MyNode)
    cursor = String()