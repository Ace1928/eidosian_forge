import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
@classmethod
def __init_subclass_with_meta__(cls, node=None, name=None, **options):
    _meta = ConnectionOptions(cls)
    base_name = re.sub('Connection$', '', name or cls.__name__) or node._meta.name
    edge_class = get_edge_class(cls, node, base_name)
    _meta.fields = {'page_info': Field(NonNull(PageInfo, name='pageInfo', required=True, description='Pagination data for this connection.')), 'edges': Field(NonNull(List(NonNull(edge_class))), description='Contains the nodes in this connection.')}
    return super(ConnectionWithNodes, cls).__init_subclass_with_meta__(node=node, name=name, _meta=_meta, **options)