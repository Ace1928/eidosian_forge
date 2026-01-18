import os
from pathlib import Path
from typing import Optional
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.names import AbstractNameDefinition, ModuleName
from jedi.inference.filters import GlobalNameFilter, ParserTreeFilter, DictFilter, MergedFilter
from jedi.inference import compiled
from jedi.inference.base_value import TreeValue
from jedi.inference.names import SubModuleName
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.compiled import create_simple_object
from jedi.inference.base_value import ValueSet
from jedi.inference.context import ModuleContext
class ModuleMixin(SubModuleDictMixin):
    _module_name_class = ModuleName

    def get_filters(self, origin_scope=None):
        yield MergedFilter(ParserTreeFilter(parent_context=self.as_context(), origin_scope=origin_scope), GlobalNameFilter(self.as_context()))
        yield DictFilter(self.sub_modules_dict())
        yield DictFilter(self._module_attributes_dict())
        yield from self.iter_star_filters()

    def py__class__(self):
        c, = values_from_qualified_names(self.inference_state, 'types', 'ModuleType')
        return c

    def is_module(self):
        return True

    def is_stub(self):
        return False

    @property
    @inference_state_method_cache()
    def name(self):
        return self._module_name_class(self, self.string_names[-1])

    @inference_state_method_cache()
    def _module_attributes_dict(self):
        names = ['__package__', '__doc__', '__name__']
        dct = dict(((n, _ModuleAttributeName(self, n)) for n in names))
        path = self.py__file__()
        if path is not None:
            dct['__file__'] = _ModuleAttributeName(self, '__file__', str(path))
        return dct

    def iter_star_filters(self):
        for star_module in self.star_imports():
            f = next(star_module.get_filters(), None)
            assert f is not None
            yield f

    @inference_state_method_cache([])
    def star_imports(self):
        from jedi.inference.imports import Importer
        modules = []
        module_context = self.as_context()
        for i in self.tree_node.iter_imports():
            if i.is_star_import():
                new = Importer(self.inference_state, import_path=i.get_paths()[-1], module_context=module_context, level=i.level).follow()
                for module in new:
                    if isinstance(module, ModuleValue):
                        modules += module.star_imports()
                modules += new
        return modules

    def get_qualified_names(self):
        """
        A module doesn't have a qualified name, but it's important to note that
        it's reachable and not `None`. With this information we can add
        qualified names on top for all value children.
        """
        return ()