from jedi import debug
from jedi.parser_utils import get_cached_parent_scope, expr_is_dotted, \
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass, \
from jedi.inference import compiled
from jedi.inference.lazy_value import LazyKnownValues, LazyTreeValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import TreeNameDefinition, ValueName
from jedi.inference.arguments import unpack_arglist, ValuesArguments
from jedi.inference.base_value import ValueSet, iterator_to_value_set, \
from jedi.inference.context import ClassContext
from jedi.inference.value.function import FunctionAndClassBase
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
from jedi.plugins import plugin_manager
class ClassValue(ClassMixin, FunctionAndClassBase, metaclass=CachedMetaClass):
    api_type = 'class'

    @inference_state_method_cache()
    def list_type_vars(self):
        found = []
        arglist = self.tree_node.get_super_arglist()
        if arglist is None:
            return []
        for stars, node in unpack_arglist(arglist):
            if stars:
                continue
            from jedi.inference.gradual.annotation import find_unknown_type_vars
            for type_var in find_unknown_type_vars(self.parent_context, node):
                if type_var not in found:
                    found.append(type_var)
        return found

    def _get_bases_arguments(self):
        arglist = self.tree_node.get_super_arglist()
        if arglist:
            from jedi.inference import arguments
            return arguments.TreeArguments(self.inference_state, self.parent_context, arglist)
        return None

    @inference_state_method_cache(default=())
    def py__bases__(self):
        args = self._get_bases_arguments()
        if args is not None:
            lst = [value for key, value in args.unpack() if key is None]
            if lst:
                return lst
        if self.py__name__() == 'object' and self.parent_context.is_builtins_module():
            return []
        return [LazyKnownValues(self.inference_state.builtins_module.py__getattribute__('object'))]

    @plugin_manager.decorate()
    def get_metaclass_filters(self, metaclasses, is_instance):
        debug.warning('Unprocessed metaclass %s', metaclasses)
        return []

    @inference_state_method_cache(default=NO_VALUES)
    def get_metaclasses(self):
        args = self._get_bases_arguments()
        if args is not None:
            m = [value for key, value in args.unpack() if key == 'metaclass']
            metaclasses = ValueSet.from_sets((lazy_value.infer() for lazy_value in m))
            metaclasses = ValueSet((m for m in metaclasses if m.is_class()))
            if metaclasses:
                return metaclasses
        for lazy_base in self.py__bases__():
            for value in lazy_base.infer():
                if value.is_class():
                    values = value.get_metaclasses()
                    if values:
                        return values
        return NO_VALUES

    @plugin_manager.decorate()
    def get_metaclass_signatures(self, metaclasses):
        return []