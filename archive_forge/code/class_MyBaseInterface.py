from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
class MyBaseInterface(Interface):
    field1 = MyScalar()