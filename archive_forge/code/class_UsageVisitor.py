from ..language.ast import (FragmentDefinition, FragmentSpread,
from ..language.visitor import ParallelVisitor, TypeInfoVisitor, Visitor, visit
from ..type import GraphQLSchema
from ..utils.type_info import TypeInfo
from .rules import specified_rules
class UsageVisitor(Visitor):
    __slots__ = ('usages', 'type_info')

    def __init__(self, usages, type_info):
        self.usages = usages
        self.type_info = type_info

    def enter_VariableDefinition(self, node, key, parent, path, ancestors):
        return False

    def enter_Variable(self, node, key, parent, path, ancestors):
        usage = VariableUsage(node, type=self.type_info.get_input_type())
        self.usages.append(usage)