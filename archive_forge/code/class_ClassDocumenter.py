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
class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):
    """
    Specialized Documenter subclass for classes.
    """
    objtype = 'class'
    member_order = 20
    option_spec: OptionSpec = {'members': members_option, 'undoc-members': bool_option, 'noindex': bool_option, 'inherited-members': inherited_members_option, 'show-inheritance': bool_option, 'member-order': member_order_option, 'exclude-members': exclude_members_option, 'private-members': members_option, 'special-members': members_option, 'class-doc-from': class_doc_from_option}
    _signature_class: Any = None
    _signature_method_name: str = None

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        if self.config.autodoc_class_signature == 'separated':
            self.options = self.options.copy()
            if self.options.special_members is None:
                self.options['special-members'] = ['__new__', '__init__']
            else:
                self.options.special_members.append('__new__')
                self.options.special_members.append('__init__')
        merge_members_option(self.options)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        return isinstance(member, type)

    def import_object(self, raiseerror: bool=False) -> bool:
        ret = super().import_object(raiseerror)
        if ret:
            if hasattr(self.object, '__name__'):
                self.doc_as_attr = self.objpath[-1] != self.object.__name__
            else:
                self.doc_as_attr = True
        return ret

    def _get_signature(self) -> Tuple[Optional[Any], Optional[str], Optional[Signature]]:

        def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
            """ Get the `attr` function or method from `obj`, if it is user-defined. """
            if inspect.is_builtin_class_method(obj, attr):
                return None
            attr = self.get_attr(obj, attr, None)
            if not (inspect.ismethod(attr) or inspect.isfunction(attr)):
                return None
            return attr
        if hasattr(self.object, '__signature__') and isinstance(self.object.__signature__, Signature):
            return (None, None, self.object.__signature__)
        call = get_user_defined_function_or_method(type(self.object), '__call__')
        if call is not None:
            if '{0.__module__}.{0.__qualname__}'.format(call) in _METACLASS_CALL_BLACKLIST:
                call = None
        if call is not None:
            self.env.app.emit('autodoc-before-process-signature', call, True)
            try:
                sig = inspect.signature(call, bound_method=True, type_aliases=self.config.autodoc_type_aliases)
                return (type(self.object), '__call__', sig)
            except ValueError:
                pass
        new = get_user_defined_function_or_method(self.object, '__new__')
        if new is not None:
            if '{0.__module__}.{0.__qualname__}'.format(new) in _CLASS_NEW_BLACKLIST:
                new = None
        if new is not None:
            self.env.app.emit('autodoc-before-process-signature', new, True)
            try:
                sig = inspect.signature(new, bound_method=True, type_aliases=self.config.autodoc_type_aliases)
                return (self.object, '__new__', sig)
            except ValueError:
                pass
        init = get_user_defined_function_or_method(self.object, '__init__')
        if init is not None:
            self.env.app.emit('autodoc-before-process-signature', init, True)
            try:
                sig = inspect.signature(init, bound_method=True, type_aliases=self.config.autodoc_type_aliases)
                return (self.object, '__init__', sig)
            except ValueError:
                pass
        self.env.app.emit('autodoc-before-process-signature', self.object, False)
        try:
            sig = inspect.signature(self.object, bound_method=False, type_aliases=self.config.autodoc_type_aliases)
            return (None, None, sig)
        except ValueError:
            pass
        return (None, None, None)

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)
        if self.config.autodoc_typehints_format == 'short':
            kwargs.setdefault('unqualified_typehints', True)
        try:
            self._signature_class, self._signature_method_name, sig = self._get_signature()
        except TypeError as exc:
            logger.warning(__('Failed to get a constructor signature for %s: %s'), self.fullname, exc)
            return None
        if sig is None:
            return None
        return stringify_signature(sig, show_return_annotation=False, **kwargs)

    def _find_signature(self) -> Tuple[str, str]:
        result = super()._find_signature()
        if result is not None:
            result = (result[0], None)
        for i, sig in enumerate(self._signatures):
            if sig.endswith(' -> None'):
                self._signatures[i] = sig[:-8]
        return result

    def format_signature(self, **kwargs: Any) -> str:
        if self.doc_as_attr:
            return ''
        if self.config.autodoc_class_signature == 'separated':
            return ''
        if self.config.autodoc_typehints_format == 'short':
            kwargs.setdefault('unqualified_typehints', True)
        sig = super().format_signature()
        sigs = []
        overloads = self.get_overloaded_signatures()
        if overloads and self.config.autodoc_typehints != 'none':
            method = safe_getattr(self._signature_class, self._signature_method_name, None)
            __globals__ = safe_getattr(method, '__globals__', {})
            for overload in overloads:
                overload = evaluate_signature(overload, __globals__, self.config.autodoc_type_aliases)
                parameters = list(overload.parameters.values())
                overload = overload.replace(parameters=parameters[1:], return_annotation=Parameter.empty)
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)
        else:
            sigs.append(sig)
        return '\n'.join(sigs)

    def get_overloaded_signatures(self) -> List[Signature]:
        if self._signature_class and self._signature_method_name:
            for cls in self._signature_class.__mro__:
                try:
                    analyzer = ModuleAnalyzer.for_module(cls.__module__)
                    analyzer.analyze()
                    qualname = '.'.join([cls.__qualname__, self._signature_method_name])
                    if qualname in analyzer.overloads:
                        return analyzer.overloads.get(qualname)
                    elif qualname in analyzer.tagorder:
                        return []
                except PycodeError:
                    pass
        return []

    def get_canonical_fullname(self) -> Optional[str]:
        __modname__ = safe_getattr(self.object, '__module__', self.modname)
        __qualname__ = safe_getattr(self.object, '__qualname__', None)
        if __qualname__ is None:
            __qualname__ = safe_getattr(self.object, '__name__', None)
        if __qualname__ and '<locals>' in __qualname__:
            __qualname__ = None
        if __modname__ and __qualname__:
            return '.'.join([__modname__, __qualname__])
        else:
            return None

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()
        if self.doc_as_attr:
            self.directivetype = 'attribute'
        super().add_directive_header(sig)
        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)
        canonical_fullname = self.get_canonical_fullname()
        if not self.doc_as_attr and canonical_fullname and (self.fullname != canonical_fullname):
            self.add_line('   :canonical: %s' % canonical_fullname, sourcename)
        if not self.doc_as_attr and self.options.show_inheritance:
            if inspect.getorigbases(self.object):
                bases = list(self.object.__orig_bases__)
            elif hasattr(self.object, '__bases__') and len(self.object.__bases__):
                bases = list(self.object.__bases__)
            else:
                bases = []
            self.env.events.emit('autodoc-process-bases', self.fullname, self.object, self.options, bases)
            if self.config.autodoc_typehints_format == 'short':
                base_classes = [restify(cls, 'smart') for cls in bases]
            else:
                base_classes = [restify(cls) for cls in bases]
            sourcename = self.get_sourcename()
            self.add_line('', sourcename)
            self.add_line('   ' + _('Bases: %s') % ', '.join(base_classes), sourcename)

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        members = get_class_members(self.object, self.objpath, self.get_attr, self.config.autodoc_inherit_docstrings)
        if not want_all:
            if not self.options.members:
                return (False, [])
            selected = []
            for name in self.options.members:
                if name in members:
                    selected.append(members[name])
                else:
                    logger.warning(__('missing attribute %s in object %s') % (name, self.fullname), type='autodoc')
            return (False, selected)
        elif self.options.inherited_members:
            return (False, list(members.values()))
        else:
            return (False, [m for m in members.values() if m.class_ == self.object])

    def get_doc(self) -> Optional[List[List[str]]]:
        if self.doc_as_attr:
            comment = self.get_variable_comment()
            if comment:
                return []
            else:
                return None
        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines
        classdoc_from = self.options.get('class-doc-from', self.config.autoclass_content)
        docstrings = []
        attrdocstring = getdoc(self.object, self.get_attr)
        if attrdocstring:
            docstrings.append(attrdocstring)
        if classdoc_from in ('both', 'init'):
            __init__ = self.get_attr(self.object, '__init__', None)
            initdocstring = getdoc(__init__, self.get_attr, self.config.autodoc_inherit_docstrings, self.object, '__init__')
            if initdocstring is not None and (initdocstring == object.__init__.__doc__ or initdocstring.strip() == object.__init__.__doc__):
                initdocstring = None
            if not initdocstring:
                __new__ = self.get_attr(self.object, '__new__', None)
                initdocstring = getdoc(__new__, self.get_attr, self.config.autodoc_inherit_docstrings, self.object, '__new__')
                if initdocstring is not None and (initdocstring == object.__new__.__doc__ or initdocstring.strip() == object.__new__.__doc__):
                    initdocstring = None
            if initdocstring:
                if classdoc_from == 'init':
                    docstrings = [initdocstring]
                else:
                    docstrings.append(initdocstring)
        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(docstring, tab_width) for docstring in docstrings]

    def get_variable_comment(self) -> Optional[List[str]]:
        try:
            key = ('', '.'.join(self.objpath))
            if self.doc_as_attr:
                analyzer = ModuleAnalyzer.for_module(self.modname)
            else:
                analyzer = ModuleAnalyzer.for_module(self.get_real_modname())
            analyzer.analyze()
            return list(analyzer.attr_docs.get(key, []))
        except PycodeError:
            return None

    def add_content(self, more_content: Optional[StringList]) -> None:
        if self.doc_as_attr and self.modname != self.get_real_modname():
            try:
                self.analyzer = ModuleAnalyzer.for_module(self.modname)
                self.analyzer.analyze()
            except PycodeError:
                pass
        if self.doc_as_attr and (not self.get_variable_comment()):
            try:
                if self.config.autodoc_typehints_format == 'short':
                    alias = restify(self.object, 'smart')
                else:
                    alias = restify(self.object)
                more_content = StringList([_('alias of %s') % alias], source='')
            except AttributeError:
                pass
        super().add_content(more_content)

    def document_members(self, all_members: bool=False) -> None:
        if self.doc_as_attr:
            return
        super().document_members(all_members)

    def generate(self, more_content: Optional[StringList]=None, real_modname: str=None, check_module: bool=False, all_members: bool=False) -> None:
        return super().generate(more_content=more_content, check_module=check_module, all_members=all_members)