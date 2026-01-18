from graphql import Undefined
from ..argument import Argument
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..objecttype import ObjectType
from ..scalars import Boolean, String
from ..schema import Schema
from ..unmountedtype import UnmountedType
from ... import NonNull
class TestUndefinedInput(InputObjectType):
    required_field = String(required=True)
    optional_field = String()