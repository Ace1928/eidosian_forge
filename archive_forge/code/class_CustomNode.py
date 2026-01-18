from graphql_relay import to_global_id
from ...types import ID, NonNull, ObjectType, String
from ...types.definitions import GrapheneObjectType
from ..node import GlobalID, Node
class CustomNode(Node):

    class Meta:
        name = 'Node'