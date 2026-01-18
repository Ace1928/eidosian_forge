from collections import OrderedDict
from textwrap import dedent
import operator
from . import ExprNodes
from . import Nodes
from . import PyrexTypes
from . import Builtin
from . import Naming
from .Errors import error, warning
from .Code import UtilityCode, TempitaUtilityCode, PyxCodeWriter
from .Visitor import VisitorTransform
from .StringEncoding import EncodedString
from .TreeFragment import TreeFragment
from .ParseTreeTransforms import NormalizeTree, SkipDeclarations
from .Options import copy_inherited_directives
def _set_up_dataclass_fields(node, fields, dataclass_module):
    variables_assignment_stats = []
    for name, field in fields.items():
        if field.private:
            continue
        for attrname in ['default', 'default_factory']:
            field_default = getattr(field, attrname)
            if field_default is MISSING or field_default.is_literal or field_default.is_name:
                continue
            global_scope = node.scope.global_scope()
            module_field_name = global_scope.mangle(global_scope.mangle(Naming.dataclass_field_default_cname, node.class_name), name)
            field_node = ExprNodes.NameNode(field_default.pos, name=EncodedString(module_field_name))
            field_node.entry = global_scope.declare_var(field_node.name, type=field_default.type or PyrexTypes.unspecified_type, pos=field_default.pos, cname=field_node.name, is_cdef=True)
            setattr(field, attrname, field_node)
            variables_assignment_stats.append(Nodes.SingleAssignmentNode(field_default.pos, lhs=field_node, rhs=field_default))
    placeholders = {}
    field_func = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('field'))
    dc_fields = ExprNodes.DictNode(node.pos, key_value_pairs=[])
    dc_fields_namevalue_assignments = []
    for name, field in fields.items():
        if field.private:
            continue
        type_placeholder_name = 'PLACEHOLDER_%s' % name
        placeholders[type_placeholder_name] = get_field_type(node.pos, node.scope.entries[name])
        field_type_placeholder_name = 'PLACEHOLDER_FIELD_TYPE_%s' % name
        if field.is_initvar:
            placeholders[field_type_placeholder_name] = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('_FIELD_INITVAR'))
        elif field.is_classvar:
            placeholders[field_type_placeholder_name] = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('_FIELD_CLASSVAR'))
        else:
            placeholders[field_type_placeholder_name] = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('_FIELD'))
        dc_field_keywords = ExprNodes.DictNode.from_pairs(node.pos, [(ExprNodes.IdentifierStringNode(node.pos, value=EncodedString(k)), FieldRecordNode(node.pos, arg=v)) for k, v in field.iterate_record_node_arguments()])
        dc_field_call = make_dataclass_call_helper(node.pos, field_func, dc_field_keywords)
        dc_fields.key_value_pairs.append(ExprNodes.DictItemNode(node.pos, key=ExprNodes.IdentifierStringNode(node.pos, value=EncodedString(name)), value=dc_field_call))
        dc_fields_namevalue_assignments.append(dedent(u'                __dataclass_fields__[{0!r}].name = {0!r}\n                __dataclass_fields__[{0!r}].type = {1}\n                __dataclass_fields__[{0!r}]._field_type = {2}\n            ').format(name, type_placeholder_name, field_type_placeholder_name))
    dataclass_fields_assignment = Nodes.SingleAssignmentNode(node.pos, lhs=ExprNodes.NameNode(node.pos, name=EncodedString('__dataclass_fields__')), rhs=dc_fields)
    dc_fields_namevalue_assignments = u'\n'.join(dc_fields_namevalue_assignments)
    dc_fields_namevalue_assignments = TreeFragment(dc_fields_namevalue_assignments, level='c_class', pipeline=[NormalizeTree(None)])
    dc_fields_namevalue_assignments = dc_fields_namevalue_assignments.substitute(placeholders)
    return variables_assignment_stats + [dataclass_fields_assignment] + dc_fields_namevalue_assignments.stats