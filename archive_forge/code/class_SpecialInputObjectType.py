from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
class SpecialInputObjectType(InputObjectType):

    @classmethod
    def __init_subclass_with_meta__(cls, other_attr='default', **options):
        _meta = SpecialInputObjectTypeOptions(cls)
        _meta.other_attr = other_attr
        super(SpecialInputObjectType, cls).__init_subclass_with_meta__(_meta=_meta, **options)