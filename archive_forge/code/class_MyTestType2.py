from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
class MyTestType2(ObjectType):

    class Meta:
        interfaces = (MyInterface,)