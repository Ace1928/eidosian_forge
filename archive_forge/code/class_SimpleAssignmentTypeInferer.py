from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
class SimpleAssignmentTypeInferer(object):
    """
    Very basic type inference.

    Note: in order to support cross-closure type inference, this must be
    applies to nested scopes in top-down order.
    """

    def set_entry_type(self, entry, entry_type, scope):
        for e in entry.all_entries():
            e.type = entry_type
            if e.type.is_memoryviewslice:
                e.init = e.type.default_value
            if e.type.is_cpp_class:
                if scope.directives['cpp_locals']:
                    e.make_cpp_optional()
                else:
                    e.type.check_nullary_constructor(entry.pos)

    def infer_types(self, scope):
        enabled = scope.directives['infer_types']
        verbose = scope.directives['infer_types.verbose']
        if enabled == True:
            spanning_type = aggressive_spanning_type
        elif enabled is None:
            spanning_type = safe_spanning_type
        else:
            for entry in scope.entries.values():
                if entry.type is unspecified_type:
                    self.set_entry_type(entry, py_object_type, scope)
            return
        assignments = set()
        assmts_resolved = set()
        dependencies = {}
        assmt_to_names = {}
        for name, entry in scope.entries.items():
            for assmt in entry.cf_assignments:
                names = assmt.type_dependencies()
                assmt_to_names[assmt] = names
                assmts = set()
                for node in names:
                    assmts.update(node.cf_state)
                dependencies[assmt] = assmts
            if entry.type is unspecified_type:
                assignments.update(entry.cf_assignments)
            else:
                assmts_resolved.update(entry.cf_assignments)

        def infer_name_node_type(node):
            types = [assmt.inferred_type for assmt in node.cf_state]
            if not types:
                node_type = py_object_type
            else:
                entry = node.entry
                node_type = spanning_type(types, entry.might_overflow, scope)
            node.inferred_type = node_type

        def infer_name_node_type_partial(node):
            types = [assmt.inferred_type for assmt in node.cf_state if assmt.inferred_type is not None]
            if not types:
                return
            entry = node.entry
            return spanning_type(types, entry.might_overflow, scope)

        def inferred_types(entry):
            has_none = False
            has_pyobjects = False
            types = []
            for assmt in entry.cf_assignments:
                if assmt.rhs.is_none:
                    has_none = True
                else:
                    rhs_type = assmt.inferred_type
                    if rhs_type and rhs_type.is_pyobject:
                        has_pyobjects = True
                    types.append(rhs_type)
            if has_none and (not has_pyobjects):
                types.append(py_object_type)
            return types

        def resolve_assignments(assignments):
            resolved = set()
            for assmt in assignments:
                deps = dependencies[assmt]
                if assmts_resolved.issuperset(deps):
                    for node in assmt_to_names[assmt]:
                        infer_name_node_type(node)
                    inferred_type = assmt.infer_type()
                    assmts_resolved.add(assmt)
                    resolved.add(assmt)
            assignments.difference_update(resolved)
            return resolved

        def partial_infer(assmt):
            partial_types = []
            for node in assmt_to_names[assmt]:
                partial_type = infer_name_node_type_partial(node)
                if partial_type is None:
                    return False
                partial_types.append((node, partial_type))
            for node, partial_type in partial_types:
                node.inferred_type = partial_type
            assmt.infer_type()
            return True
        partial_assmts = set()

        def resolve_partial(assignments):
            partials = set()
            for assmt in assignments:
                if assmt in partial_assmts:
                    continue
                if partial_infer(assmt):
                    partials.add(assmt)
                    assmts_resolved.add(assmt)
            partial_assmts.update(partials)
            return partials
        while True:
            if not resolve_assignments(assignments):
                if not resolve_partial(assignments):
                    break
        inferred = set()
        for entry in scope.entries.values():
            if entry.type is not unspecified_type:
                continue
            entry_type = py_object_type
            if assmts_resolved.issuperset(entry.cf_assignments):
                types = inferred_types(entry)
                if types and all(types):
                    entry_type = spanning_type(types, entry.might_overflow, scope)
                    inferred.add(entry)
            self.set_entry_type(entry, entry_type, scope)

        def reinfer():
            dirty = False
            for entry in inferred:
                for assmt in entry.cf_assignments:
                    assmt.infer_type()
                types = inferred_types(entry)
                new_type = spanning_type(types, entry.might_overflow, scope)
                if new_type != entry.type:
                    self.set_entry_type(entry, new_type, scope)
                    dirty = True
            return dirty
        while reinfer():
            pass
        if verbose:
            for entry in inferred:
                message(entry.pos, "inferred '%s' to be of type '%s'" % (entry.name, entry.type))