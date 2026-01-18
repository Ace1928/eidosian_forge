import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
class BaseEdge:
    extra = String()