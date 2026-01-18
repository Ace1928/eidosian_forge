import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
class MyObjectConnection(Connection):

    class Meta:
        node = MyObject
        strict_types = True