from typing import Any
from ...error import GraphQLError
from ...language import (
from ...type import is_composite_type
from ...utilities import type_from_ast
from . import ValidationRule
Fragments on composite type

    Fragments use a type condition to determine if they apply, since fragments can only
    be spread into a composite type (object, interface, or union), the type condition
    must also be a composite type.

    See https://spec.graphql.org/draft/#sec-Fragments-On-Composite-Types
    