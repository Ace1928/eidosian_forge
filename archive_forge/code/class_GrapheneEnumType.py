from enum import Enum as PyEnum
from graphql import (
class GrapheneEnumType(GrapheneGraphQLType, GraphQLEnumType):

    def serialize(self, value):
        if not isinstance(value, PyEnum):
            enum = self.graphene_type._meta.enum
            try:
                value = enum(value)
            except ValueError:
                try:
                    value = enum[value]
                except KeyError:
                    pass
        return super(GrapheneEnumType, self).serialize(value)