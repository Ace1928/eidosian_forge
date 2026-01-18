import re
from collections.abc import Iterable
from functools import partial
from typing import Type
from graphql_relay import connection_from_array
from ..types import Boolean, Enum, Int, Interface, List, NonNull, Scalar, String, Union
from ..types.field import Field
from ..types.objecttype import ObjectType, ObjectTypeOptions
from ..utils.thenables import maybe_thenable
from .node import is_node, AbstractNode
class ConnectionOptions(ObjectTypeOptions):
    node = None