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
class ClassMixin:

    def is_class(self):
        return True

    def is_class_mixin(self):
        return True

    def py__call__(self, arguments):
        from jedi.inference.value import TreeInstance
        from jedi.inference.gradual.typing import TypedDict
        if self.is_typeddict():
            return ValueSet([TypedDict(self)])
        return ValueSet([TreeInstance(self.inference_state, self.parent_context, self, arguments)])

    def py__class__(self):
        return compiled.builtin_from_name(self.inference_state, 'type')

    @property
    def name(self):
        return ValueName(self, self.tree_node.name)

    def py__name__(self):
        return self.name.string_name

    @inference_state_method_generator_cache()
    def py__mro__(self):
        mro = [self]
        yield self
        for lazy_cls in self.py__bases__():
            for cls in lazy_cls.infer():
                try:
                    mro_method = cls.py__mro__
                except AttributeError:
                    '\n                    >>> class Y(lambda: test): pass\n                    Traceback (most recent call last):\n                      File "<stdin>", line 1, in <module>\n                    TypeError: function() argument 1 must be code, not str\n                    >>> class Y(1): pass\n                    Traceback (most recent call last):\n                      File "<stdin>", line 1, in <module>\n                    TypeError: int() takes at most 2 arguments (3 given)\n                    '
                    debug.warning('Super class of %s is not a class: %s', self, cls)
                else:
                    for cls_new in mro_method():
                        if cls_new not in mro:
                            mro.append(cls_new)
                            yield cls_new

    def get_filters(self, origin_scope=None, is_instance=False, include_metaclasses=True, include_type_when_class=True):
        if include_metaclasses:
            metaclasses = self.get_metaclasses()
            if metaclasses:
                yield from self.get_metaclass_filters(metaclasses, is_instance)
        for cls in self.py__mro__():
            if cls.is_compiled():
                yield from cls.get_filters(is_instance=is_instance)
            else:
                yield ClassFilter(self, node_context=cls.as_context(), origin_scope=origin_scope, is_instance=is_instance)
        if not is_instance and include_type_when_class:
            from jedi.inference.compiled import builtin_from_name
            type_ = builtin_from_name(self.inference_state, 'type')
            assert isinstance(type_, ClassValue)
            if type_ != self:
                args = ValuesArguments([])
                for instance in type_.py__call__(args):
                    instance_filters = instance.get_filters()
                    next(instance_filters, None)
                    next(instance_filters, None)
                    x = next(instance_filters, None)
                    assert x is not None
                    yield x

    def get_signatures(self):
        metaclasses = self.get_metaclasses()
        if metaclasses:
            sigs = self.get_metaclass_signatures(metaclasses)
            if sigs:
                return sigs
        args = ValuesArguments([])
        init_funcs = self.py__call__(args).py__getattribute__('__init__')
        return [sig.bind(self) for sig in init_funcs.get_signatures()]

    def _as_context(self):
        return ClassContext(self)

    def get_type_hint(self, add_class_info=True):
        if add_class_info:
            return 'Type[%s]' % self.py__name__()
        return self.py__name__()

    @inference_state_method_cache(default=False)
    def is_typeddict(self):
        from jedi.inference.gradual.typing import TypedDictClass
        for lazy_cls in self.py__bases__():
            if not isinstance(lazy_cls, LazyTreeValue):
                return False
            tree_node = lazy_cls.data
            if not expr_is_dotted(tree_node):
                return False
            for cls in lazy_cls.infer():
                if isinstance(cls, TypedDictClass):
                    return True
                try:
                    method = cls.is_typeddict
                except AttributeError:
                    return False
                else:
                    if method():
                        return True
        return False

    def py__getitem__(self, index_value_set, contextualized_node):
        from jedi.inference.gradual.base import GenericClass
        if not index_value_set:
            debug.warning('Class indexes inferred to nothing. Returning class instead')
            return ValueSet([self])
        return ValueSet((GenericClass(self, LazyGenericManager(context_of_index=contextualized_node.context, index_value=index_value)) for index_value in index_value_set))

    def with_generics(self, generics_tuple):
        from jedi.inference.gradual.base import GenericClass
        return GenericClass(self, TupleGenericManager(generics_tuple))

    def define_generics(self, type_var_dict):
        from jedi.inference.gradual.base import GenericClass

        def remap_type_vars():
            """
            The TypeVars in the resulting classes have sometimes different names
            and we need to check for that, e.g. a signature can be:

            def iter(iterable: Iterable[_T]) -> Iterator[_T]: ...

            However, the iterator is defined as Iterator[_T_co], which means it has
            a different type var name.
            """
            for type_var in self.list_type_vars():
                yield type_var_dict.get(type_var.py__name__(), NO_VALUES)
        if type_var_dict:
            return ValueSet([GenericClass(self, TupleGenericManager(tuple(remap_type_vars())))])
        return ValueSet({self})