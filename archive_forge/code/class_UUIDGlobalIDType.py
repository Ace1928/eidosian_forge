from graphql_relay import from_global_id, to_global_id
from ..types import ID, UUID
from ..types.base import BaseType
from typing import Type
class UUIDGlobalIDType(BaseGlobalIDType):
    """
    UUID global ID type.
    By definition UUID are global so they are used as they are.
    """
    graphene_type = UUID

    @classmethod
    def resolve_global_id(cls, info, global_id):
        _type = info.return_type.graphene_type._meta.name
        return (_type, global_id)

    @classmethod
    def to_global_id(cls, _type, _id):
        return _id