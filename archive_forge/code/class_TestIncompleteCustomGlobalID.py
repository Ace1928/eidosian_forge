import re
from uuid import uuid4
from graphql import graphql_sync
from ..id_type import BaseGlobalIDType, SimpleGlobalIDType, UUIDGlobalIDType
from ..node import Node
from ...types import Int, ObjectType, Schema, String
class TestIncompleteCustomGlobalID:

    def setup(self):
        self.user_list = [{'id': 1, 'name': 'First'}, {'id': 2, 'name': 'Second'}, {'id': 3, 'name': 'Third'}, {'id': 4, 'name': 'Fourth'}]
        self.users = {user['id']: user for user in self.user_list}

    def test_must_define_to_global_id(self):
        """
        Test that if the `to_global_id` method is not defined, we can query the object, but we can't request its ID.
        """

        class CustomGlobalIDType(BaseGlobalIDType):
            graphene_type = Int

            @classmethod
            def resolve_global_id(cls, info, global_id):
                _type = info.return_type.graphene_type._meta.name
                return (_type, global_id)

        class CustomNode(Node):

            class Meta:
                global_id_type = CustomGlobalIDType

        class User(ObjectType):

            class Meta:
                interfaces = [CustomNode]
            name = String()

            @classmethod
            def get_node(cls, _type, _id):
                return self.users[_id]

        class RootQuery(ObjectType):
            user = CustomNode.Field(User)
        self.schema = Schema(query=RootQuery, types=[User])
        self.graphql_schema = self.schema.graphql_schema
        query = 'query {\n            user(id: 2) {\n                name\n            }\n        }'
        result = graphql_sync(self.graphql_schema, query)
        assert not result.errors
        assert result.data['user']['name'] == self.user_list[1]['name']
        query = 'query {\n            user(id: 2) {\n                id\n                name\n            }\n        }'
        result = graphql_sync(self.graphql_schema, query)
        assert result.errors is not None
        assert len(result.errors) == 1
        assert result.errors[0].path == ['user', 'id']

    def test_must_define_resolve_global_id(self):
        """
        Test that if the `resolve_global_id` method is not defined, we can't query the object by ID.
        """

        class CustomGlobalIDType(BaseGlobalIDType):
            graphene_type = Int

            @classmethod
            def to_global_id(cls, _type, _id):
                return _id

        class CustomNode(Node):

            class Meta:
                global_id_type = CustomGlobalIDType

        class User(ObjectType):

            class Meta:
                interfaces = [CustomNode]
            name = String()

            @classmethod
            def get_node(cls, _type, _id):
                return self.users[_id]

        class RootQuery(ObjectType):
            user = CustomNode.Field(User)
        self.schema = Schema(query=RootQuery, types=[User])
        self.graphql_schema = self.schema.graphql_schema
        query = 'query {\n            user(id: 2) {\n                id\n                name\n            }\n        }'
        result = graphql_sync(self.graphql_schema, query)
        assert result.errors is not None
        assert len(result.errors) == 1
        assert result.errors[0].path == ['user']