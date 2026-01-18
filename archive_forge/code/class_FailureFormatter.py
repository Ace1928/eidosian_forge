import functools
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import tree
from taskflow.utils import misc
class FailureFormatter(object):
    """Formats a failure and connects it to associated atoms & engine."""
    _BUILDERS = {states.EXECUTE: (_fetch_predecessor_tree, 'predecessors')}

    def __init__(self, engine, hide_inputs_outputs_of=()):
        self._hide_inputs_outputs_of = hide_inputs_outputs_of
        self._engine = engine

    def _format_node(self, storage, cache, node):
        """Formats a single tree node into a string version."""
        if node.metadata['kind'] == compiler.FLOW:
            flow = node.item
            flow_name = flow.name
            return "Flow '%s'" % flow_name
        elif node.metadata['kind'] in compiler.ATOMS:
            atom = node.item
            atom_name = atom.name
            atom_attrs = {}
            intention, intention_found = _cached_get(cache, 'intentions', atom_name, storage.get_atom_intention, atom_name)
            if intention_found:
                atom_attrs['intention'] = intention
            state, state_found = _cached_get(cache, 'states', atom_name, storage.get_atom_state, atom_name)
            if state_found:
                atom_attrs['state'] = state
            if atom_name not in self._hide_inputs_outputs_of:
                fetch_mapped_args = functools.partial(storage.fetch_mapped_args, atom.rebind, atom_name=atom_name, optional_args=atom.optional)
                requires, requires_found = _cached_get(cache, 'requires', atom_name, fetch_mapped_args)
                if requires_found:
                    atom_attrs['requires'] = requires
                provides, provides_found = _cached_get(cache, 'provides', atom_name, storage.get_execute_result, atom_name)
                if provides_found:
                    atom_attrs['provides'] = provides
            if atom_attrs:
                return "Atom '%s' %s" % (atom_name, atom_attrs)
            else:
                return "Atom '%s'" % atom_name
        else:
            raise TypeError("Unable to format node, unknown node kind '%s' encountered" % node.metadata['kind'])

    def format(self, fail, atom_matcher):
        """Returns a (exc_info, details) tuple about the failure.

        The ``exc_info`` tuple should be a standard three element
        (exctype, value, traceback) tuple that will be used for further
        logging. A non-empty string is typically returned for ``details``; it
        should contain any string info about the failure (with any specific
        details the ``exc_info`` may not have/contain).
        """
        buff = misc.StringIO()
        storage = self._engine.storage
        compilation = self._engine.compilation
        if fail.exc_info is None:
            buff.write_nl(fail.pformat(traceback=True))
        if storage is None or compilation is None:
            return (fail.exc_info, buff.getvalue())
        hierarchy = compilation.hierarchy
        graph = compilation.execution_graph
        atom_node = hierarchy.find_first_match(atom_matcher)
        atom = None
        atom_intention = None
        if atom_node is not None:
            atom = atom_node.item
            atom_intention = storage.get_atom_intention(atom.name)
        if atom is not None and atom_intention in self._BUILDERS:
            cache = {'intentions': {}, 'provides': {}, 'requires': {}, 'states': {}}
            builder, kind = self._BUILDERS[atom_intention]
            rooted_tree = builder(graph, atom)
            child_count = rooted_tree.child_count(only_direct=False)
            buff.write_nl('%s %s (most recent first):' % (child_count, kind))
            formatter = functools.partial(self._format_node, storage, cache)
            direct_child_count = rooted_tree.child_count(only_direct=True)
            for i, child in enumerate(rooted_tree, 1):
                if i == direct_child_count:
                    buff.write(child.pformat(stringify_node=formatter, starting_prefix='  '))
                else:
                    buff.write_nl(child.pformat(stringify_node=formatter, starting_prefix='  '))
        return (fail.exc_info, buff.getvalue())