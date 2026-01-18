import re
from uuid import uuid4
from graphql import graphql_sync
from ..id_type import BaseGlobalIDType, SimpleGlobalIDType, UUIDGlobalIDType
from ..node import Node
from ...types import Int, ObjectType, Schema, String
class TestUUIDGlobalID:

    def setup(self):
        self.user_list = [{'id': uuid4(), 'name': 'First'}, {'id': uuid4(), 'name': 'Second'}, {'id': uuid4(), 'name': 'Third'}, {'id': uuid4(), 'name': 'Fourth'}]
        self.users = {user['id']: user for user in self.user_list}

        class CustomNode(Node):

            class Meta:
                global_id_type = UUIDGlobalIDType

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

    def test_str_schema_correct(self):
        """
        Check that the schema has the expected and custom node interface and user type and that they both use UUIDs
        """
        parsed = re.findall('(.+) \\{\\n\\s*([\\w\\W]*?)\\n\\}', str(self.schema))
        types = [t for t, f in parsed]
        fields = [f for t, f in parsed]
        custom_node_interface = 'interface CustomNode'
        assert custom_node_interface in types
        assert '"""The ID of the object"""\n  id: UUID!' == fields[types.index(custom_node_interface)]
        user_type = 'type User implements CustomNode'
        assert user_type in types
        assert '"""The ID of the object"""\n  id: UUID!\n  name: String' == fields[types.index(user_type)]

    def test_get_by_id(self):
        query = 'query userById($id: UUID!) {\n            user(id: $id) {\n                id\n                name\n            }\n        }'
        result = graphql_sync(self.graphql_schema, query, variable_values={'id': str(self.user_list[0]['id'])})
        assert not result.errors
        assert result.data['user']['id'] == str(self.user_list[0]['id'])
        assert result.data['user']['name'] == self.user_list[0]['name']