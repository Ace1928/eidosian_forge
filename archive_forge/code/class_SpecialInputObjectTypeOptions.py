from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
class SpecialInputObjectTypeOptions(ObjectTypeOptions):
    other_attr = None