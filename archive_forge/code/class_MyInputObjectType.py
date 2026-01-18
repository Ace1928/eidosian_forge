from graphql import Undefined
from graphql.type import (
from ..dynamic import Dynamic
from ..enum import Enum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Int, String
from ..schema import Schema
from ..structures import List, NonNull
class MyInputObjectType(InputObjectType):
    """Description"""
    foo_bar = String(description='Field description')
    bar = String(name='gizmo')
    baz = NonNull(MyInnerObjectType)
    own = InputField(lambda: MyInputObjectType)

    def resolve_foo_bar(self, args, info):
        return args.get('bar')