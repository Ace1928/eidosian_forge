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
class ModuleDocumenter(Documenter):
    """
    Specialized Documenter subclass for modules.
    """
    objtype = 'module'
    content_indent = ''
    titles_allowed = True
    _extra_indent = '   '
    option_spec: OptionSpec = {'members': members_option, 'undoc-members': bool_option, 'noindex': bool_option, 'inherited-members': inherited_members_option, 'show-inheritance': bool_option, 'synopsis': identity, 'platform': identity, 'deprecated': bool_option, 'member-order': member_order_option, 'exclude-members': exclude_members_option, 'private-members': members_option, 'special-members': members_option, 'imported-members': bool_option, 'ignore-module-all': bool_option, 'no-value': bool_option}

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        merge_members_option(self.options)
        self.__all__: Optional[Sequence[str]] = None

    def add_content(self, more_content: Optional[StringList]) -> None:
        old_indent = self.indent
        self.indent += self._extra_indent
        super().add_content(None)
        self.indent = old_indent
        if more_content:
            for line, src in zip(more_content.data, more_content.items):
                self.add_line(line, src[0], src[1])

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        return False

    def resolve_name(self, modname: str, parents: Any, path: str, base: Any) -> Tuple[str, List[str]]:
        if modname is not None:
            logger.warning(__('"::" in automodule name doesn\'t make sense'), type='autodoc')
        return ((path or '') + base, [])

    def parse_name(self) -> bool:
        ret = super().parse_name()
        if self.args or self.retann:
            logger.warning(__('signature arguments or return annotation given for automodule %s') % self.fullname, type='autodoc')
        return ret

    def import_object(self, raiseerror: bool=False) -> bool:
        ret = super().import_object(raiseerror)
        try:
            if not self.options.ignore_module_all:
                self.__all__ = inspect.getall(self.object)
        except ValueError as exc:
            logger.warning(__('__all__ should be a list of strings, not %r (in module %s) -- ignoring __all__') % (exc.args[0], self.fullname), type='autodoc')
        return ret

    def add_directive_header(self, sig: str) -> None:
        Documenter.add_directive_header(self, sig)
        sourcename = self.get_sourcename()
        if self.options.synopsis:
            self.add_line('   :synopsis: ' + self.options.synopsis, sourcename)
        if self.options.platform:
            self.add_line('   :platform: ' + self.options.platform, sourcename)
        if self.options.deprecated:
            self.add_line('   :deprecated:', sourcename)

    def get_module_members(self) -> Dict[str, ObjectMember]:
        """Get members of target module."""
        if self.analyzer:
            attr_docs = self.analyzer.attr_docs
        else:
            attr_docs = {}
        members: Dict[str, ObjectMember] = {}
        for name in dir(self.object):
            try:
                value = safe_getattr(self.object, name, None)
                if ismock(value):
                    value = undecorate(value)
                docstring = attr_docs.get(('', name), [])
                members[name] = ObjectMember(name, value, docstring='\n'.join(docstring))
            except AttributeError:
                continue
        for name in inspect.getannotations(self.object):
            if name not in members:
                docstring = attr_docs.get(('', name), [])
                members[name] = ObjectMember(name, INSTANCEATTR, docstring='\n'.join(docstring))
        return members

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        members = self.get_module_members()
        if want_all:
            if self.__all__ is None:
                return (True, list(members.values()))
            else:
                for member in members.values():
                    if member.__name__ not in self.__all__:
                        member.skipped = True
                return (False, list(members.values()))
        else:
            memberlist = self.options.members or []
            ret = []
            for name in memberlist:
                if name in members:
                    ret.append(members[name])
                else:
                    logger.warning(__('missing attribute mentioned in :members: option: module %s, attribute %s') % (safe_getattr(self.object, '__name__', '???'), name), type='autodoc')
            return (False, ret)

    def sort_members(self, documenters: List[Tuple['Documenter', bool]], order: str) -> List[Tuple['Documenter', bool]]:
        if order == 'bysource' and self.__all__:
            documenters.sort(key=lambda e: e[0].name)

            def keyfunc(entry: Tuple[Documenter, bool]) -> int:
                name = entry[0].name.split('::')[1]
                if self.__all__ and name in self.__all__:
                    return self.__all__.index(name)
                else:
                    return len(self.__all__)
            documenters.sort(key=keyfunc)
            return documenters
        else:
            return super().sort_members(documenters, order)