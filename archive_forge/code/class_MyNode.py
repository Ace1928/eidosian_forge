from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
class MyNode(ObjectType):
    id = ID()
    name = String()