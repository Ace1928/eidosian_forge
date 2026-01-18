import re
from uuid import uuid4
from graphql import graphql_sync
from ..id_type import BaseGlobalIDType, SimpleGlobalIDType, UUIDGlobalIDType
from ..node import Node
from ...types import Int, ObjectType, Schema, String
class CustomGlobalIDType(BaseGlobalIDType):
    graphene_type = Int

    @classmethod
    def to_global_id(cls, _type, _id):
        return _id