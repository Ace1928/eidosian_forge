from functools import partial
from inspect import isclass
from ..types import Field, Interface, ObjectType
from ..types.interface import InterfaceOptions
from ..types.utils import get_type
from .id_type import BaseGlobalIDType, DefaultGlobalIDType
class GlobalID(Field):

    def __init__(self, node=None, parent_type=None, required=True, global_id_type=DefaultGlobalIDType, *args, **kwargs):
        super(GlobalID, self).__init__(global_id_type.graphene_type, *args, required=required, **kwargs)
        self.node = node or Node
        self.parent_type_name = parent_type._meta.name if parent_type else None

    @staticmethod
    def id_resolver(parent_resolver, node, root, info, parent_type_name=None, **args):
        type_id = parent_resolver(root, info, **args)
        parent_type_name = parent_type_name or info.parent_type.name
        return node.to_global_id(parent_type_name, type_id)

    def wrap_resolve(self, parent_resolver):
        return partial(self.id_resolver, parent_resolver, self.node, parent_type_name=self.parent_type_name)