import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence,
from docutils.statemachine import StringList
import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import (get_class_members, get_object_members, import_module,
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
from sphinx.util.typing import OptionSpec, get_type_hints, restify
from sphinx.util.typing import stringify as stringify_typehint
class FunctionDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):
    """
    Specialized Documenter subclass for functions.
    """
    objtype = 'function'
    member_order = 30

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        return inspect.isfunction(member) or inspect.isbuiltin(member) or (inspect.isroutine(member) and isinstance(parent, ModuleDocumenter))

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)
        if self.config.autodoc_typehints_format == 'short':
            kwargs.setdefault('unqualified_typehints', True)
        try:
            self.env.app.emit('autodoc-before-process-signature', self.object, False)
            sig = inspect.signature(self.object, type_aliases=self.config.autodoc_type_aliases)
            args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            logger.warning(__('Failed to get a function signature for %s: %s'), self.fullname, exc)
            return None
        except ValueError:
            args = ''
        if self.config.strip_signature_backslash:
            args = args.replace('\\', '\\\\')
        return args

    def document_members(self, all_members: bool=False) -> None:
        pass

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()
        super().add_directive_header(sig)
        if inspect.iscoroutinefunction(self.object) or inspect.isasyncgenfunction(self.object):
            self.add_line('   :async:', sourcename)

    def format_signature(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints_format == 'short':
            kwargs.setdefault('unqualified_typehints', True)
        sigs = []
        if self.analyzer and '.'.join(self.objpath) in self.analyzer.overloads and (self.config.autodoc_typehints != 'none'):
            overloaded = True
        else:
            overloaded = False
            sig = super().format_signature(**kwargs)
            sigs.append(sig)
        if inspect.is_singledispatch_function(self.object):
            for typ, func in self.object.registry.items():
                if typ is object:
                    pass
                else:
                    dispatchfunc = self.annotate_to_first_argument(func, typ)
                    if dispatchfunc:
                        documenter = FunctionDocumenter(self.directive, '')
                        documenter.object = dispatchfunc
                        documenter.objpath = [None]
                        sigs.append(documenter.format_signature())
        if overloaded:
            actual = inspect.signature(self.object, type_aliases=self.config.autodoc_type_aliases)
            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = self.merge_default_value(actual, overload)
                overload = evaluate_signature(overload, __globals__, self.config.autodoc_type_aliases)
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)
        return '\n'.join(sigs)

    def merge_default_value(self, actual: Signature, overload: Signature) -> Signature:
        """Merge default values of actual implementation to the overload variants."""
        parameters = list(overload.parameters.values())
        for i, param in enumerate(parameters):
            actual_param = actual.parameters.get(param.name)
            if actual_param and param.default == '...':
                parameters[i] = param.replace(default=actual_param.default)
        return overload.replace(parameters=parameters)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> Optional[Callable]:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
        except TypeError as exc:
            logger.warning(__('Failed to get a function signature for %s: %s'), self.fullname, exc)
            return None
        except ValueError:
            return None
        if len(sig.parameters) == 0:
            return None

        def dummy():
            pass
        params = list(sig.parameters.values())
        if params[0].annotation is Parameter.empty:
            params[0] = params[0].replace(annotation=typ)
            try:
                dummy.__signature__ = sig.replace(parameters=params)
                return dummy
            except (AttributeError, TypeError):
                return None
        return func