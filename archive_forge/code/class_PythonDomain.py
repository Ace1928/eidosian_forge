import builtins
import inspect
import re
import sys
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref, pending_xref_condition
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import (find_pending_xref_condition, make_id, make_refnode,
from sphinx.util.typing import OptionSpec, TextlikeNode
class PythonDomain(Domain):
    """Python language domain."""
    name = 'py'
    label = 'Python'
    object_types: Dict[str, ObjType] = {'function': ObjType(_('function'), 'func', 'obj'), 'data': ObjType(_('data'), 'data', 'obj'), 'class': ObjType(_('class'), 'class', 'exc', 'obj'), 'exception': ObjType(_('exception'), 'exc', 'class', 'obj'), 'method': ObjType(_('method'), 'meth', 'obj'), 'classmethod': ObjType(_('class method'), 'meth', 'obj'), 'staticmethod': ObjType(_('static method'), 'meth', 'obj'), 'attribute': ObjType(_('attribute'), 'attr', 'obj'), 'property': ObjType(_('property'), 'attr', '_prop', 'obj'), 'module': ObjType(_('module'), 'mod', 'obj')}
    directives = {'function': PyFunction, 'data': PyVariable, 'class': PyClasslike, 'exception': PyClasslike, 'method': PyMethod, 'classmethod': PyClassMethod, 'staticmethod': PyStaticMethod, 'attribute': PyAttribute, 'property': PyProperty, 'module': PyModule, 'currentmodule': PyCurrentModule, 'decorator': PyDecoratorFunction, 'decoratormethod': PyDecoratorMethod}
    roles = {'data': PyXRefRole(), 'exc': PyXRefRole(), 'func': PyXRefRole(fix_parens=True), 'class': PyXRefRole(), 'const': PyXRefRole(), 'attr': PyXRefRole(), 'meth': PyXRefRole(fix_parens=True), 'mod': PyXRefRole(), 'obj': PyXRefRole()}
    initial_data: Dict[str, Dict[str, Tuple[Any]]] = {'objects': {}, 'modules': {}}
    indices = [PythonModuleIndex]

    @property
    def objects(self) -> Dict[str, ObjectEntry]:
        return self.data.setdefault('objects', {})

    def note_object(self, name: str, objtype: str, node_id: str, aliased: bool=False, location: Any=None) -> None:
        """Note a python object for cross reference.

        .. versionadded:: 2.1
        """
        if name in self.objects:
            other = self.objects[name]
            if other.aliased and aliased is False:
                pass
            elif other.aliased is False and aliased:
                return
            else:
                logger.warning(__('duplicate object description of %s, other instance in %s, use :noindex: for one of them'), name, other.docname, location=location)
        self.objects[name] = ObjectEntry(self.env.docname, node_id, objtype, aliased)

    @property
    def modules(self) -> Dict[str, ModuleEntry]:
        return self.data.setdefault('modules', {})

    def note_module(self, name: str, node_id: str, synopsis: str, platform: str, deprecated: bool) -> None:
        """Note a python module for cross reference.

        .. versionadded:: 2.1
        """
        self.modules[name] = ModuleEntry(self.env.docname, node_id, synopsis, platform, deprecated)

    def clear_doc(self, docname: str) -> None:
        for fullname, obj in list(self.objects.items()):
            if obj.docname == docname:
                del self.objects[fullname]
        for modname, mod in list(self.modules.items()):
            if mod.docname == docname:
                del self.modules[modname]

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        for fullname, obj in otherdata['objects'].items():
            if obj.docname in docnames:
                self.objects[fullname] = obj
        for modname, mod in otherdata['modules'].items():
            if mod.docname in docnames:
                self.modules[modname] = mod

    def find_obj(self, env: BuildEnvironment, modname: str, classname: str, name: str, type: str, searchmode: int=0) -> List[Tuple[str, ObjectEntry]]:
        """Find a Python object for "name", perhaps using the given module
        and/or classname.  Returns a list of (name, object entry) tuples.
        """
        if name[-2:] == '()':
            name = name[:-2]
        if not name:
            return []
        matches: List[Tuple[str, ObjectEntry]] = []
        newname = None
        if searchmode == 1:
            if type is None:
                objtypes = list(self.object_types)
            else:
                objtypes = self.objtypes_for_role(type)
            if objtypes is not None:
                if modname and classname:
                    fullname = modname + '.' + classname + '.' + name
                    if fullname in self.objects and self.objects[fullname].objtype in objtypes:
                        newname = fullname
                if not newname:
                    if modname and modname + '.' + name in self.objects and (self.objects[modname + '.' + name].objtype in objtypes):
                        newname = modname + '.' + name
                    elif name in self.objects and self.objects[name].objtype in objtypes:
                        newname = name
                    else:
                        searchname = '.' + name
                        matches = [(oname, self.objects[oname]) for oname in self.objects if oname.endswith(searchname) and self.objects[oname].objtype in objtypes]
        elif name in self.objects:
            newname = name
        elif type == 'mod':
            return []
        elif classname and classname + '.' + name in self.objects:
            newname = classname + '.' + name
        elif modname and modname + '.' + name in self.objects:
            newname = modname + '.' + name
        elif modname and classname and (modname + '.' + classname + '.' + name in self.objects):
            newname = modname + '.' + classname + '.' + name
        if newname is not None:
            matches.append((newname, self.objects[newname]))
        return matches

    def resolve_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, type: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        modname = node.get('py:module')
        clsname = node.get('py:class')
        searchmode = 1 if node.hasattr('refspecific') else 0
        matches = self.find_obj(env, modname, clsname, target, type, searchmode)
        if not matches and type == 'attr':
            matches = self.find_obj(env, modname, clsname, target, 'meth', searchmode)
        if not matches and type == 'meth':
            matches = self.find_obj(env, modname, clsname, target, '_prop', searchmode)
        if not matches:
            return None
        elif len(matches) > 1:
            canonicals = [m for m in matches if not m[1].aliased]
            if len(canonicals) == 1:
                matches = canonicals
            else:
                logger.warning(__('more than one target found for cross-reference %r: %s'), target, ', '.join((match[0] for match in matches)), type='ref', subtype='python', location=node)
        name, obj = matches[0]
        if obj[2] == 'module':
            return self._make_module_refnode(builder, fromdocname, name, contnode)
        else:
            content = find_pending_xref_condition(node, 'resolved')
            if content:
                children = content.children
            else:
                children = [contnode]
            return make_refnode(builder, fromdocname, obj[0], obj[1], children, name)

    def resolve_any_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, target: str, node: pending_xref, contnode: Element) -> List[Tuple[str, Element]]:
        modname = node.get('py:module')
        clsname = node.get('py:class')
        results: List[Tuple[str, Element]] = []
        matches = self.find_obj(env, modname, clsname, target, None, 1)
        multiple_matches = len(matches) > 1
        for name, obj in matches:
            if multiple_matches and obj.aliased:
                continue
            if obj[2] == 'module':
                results.append(('py:mod', self._make_module_refnode(builder, fromdocname, name, contnode)))
            else:
                content = find_pending_xref_condition(node, 'resolved')
                if content:
                    children = content.children
                else:
                    children = [contnode]
                results.append(('py:' + self.role_for_objtype(obj[2]), make_refnode(builder, fromdocname, obj[0], obj[1], children, name)))
        return results

    def _make_module_refnode(self, builder: Builder, fromdocname: str, name: str, contnode: Node) -> Element:
        module = self.modules[name]
        title = name
        if module.synopsis:
            title += ': ' + module.synopsis
        if module.deprecated:
            title += _(' (deprecated)')
        if module.platform:
            title += ' (' + module.platform + ')'
        return make_refnode(builder, fromdocname, module.docname, module.node_id, contnode, title)

    def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
        for modname, mod in self.modules.items():
            yield (modname, modname, 'module', mod.docname, mod.node_id, 0)
        for refname, obj in self.objects.items():
            if obj.objtype != 'module':
                if obj.aliased:
                    yield (refname, refname, obj.objtype, obj.docname, obj.node_id, -1)
                else:
                    yield (refname, refname, obj.objtype, obj.docname, obj.node_id, 1)

    def get_full_qualified_name(self, node: Element) -> Optional[str]:
        modname = node.get('py:module')
        clsname = node.get('py:class')
        target = node.get('reftarget')
        if target is None:
            return None
        else:
            return '.'.join(filter(None, [modname, clsname, target]))