import re
import sys
from copy import copy
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional,
from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.locale import _, __
from sphinx.roles import EmphasizedLiteral, XRefRole
from sphinx.util import docname_join, logging, ws_re
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import clean_astext, make_id, make_refnode
from sphinx.util.typing import OptionSpec, RoleFunction
class StandardDomain(Domain):
    """
    Domain for all objects that don't fit into another domain or are added
    via the application interface.
    """
    name = 'std'
    label = 'Default'
    object_types: Dict[str, ObjType] = {'term': ObjType(_('glossary term'), 'term', searchprio=-1), 'token': ObjType(_('grammar token'), 'token', searchprio=-1), 'label': ObjType(_('reference label'), 'ref', 'keyword', searchprio=-1), 'envvar': ObjType(_('environment variable'), 'envvar'), 'cmdoption': ObjType(_('program option'), 'option'), 'doc': ObjType(_('document'), 'doc', searchprio=-1)}
    directives: Dict[str, Type[Directive]] = {'program': Program, 'cmdoption': Cmdoption, 'option': Cmdoption, 'envvar': EnvVar, 'glossary': Glossary, 'productionlist': ProductionList}
    roles: Dict[str, Union[RoleFunction, XRefRole]] = {'option': OptionXRefRole(warn_dangling=True), 'envvar': EnvVarXRefRole(), 'token': TokenXRefRole(), 'term': XRefRole(innernodeclass=nodes.inline, warn_dangling=True), 'ref': XRefRole(lowercase=True, innernodeclass=nodes.inline, warn_dangling=True), 'numref': XRefRole(lowercase=True, warn_dangling=True), 'keyword': XRefRole(warn_dangling=True), 'doc': XRefRole(warn_dangling=True, innernodeclass=nodes.inline)}
    initial_data: Final = {'progoptions': {}, 'objects': {}, 'labels': {'genindex': ('genindex', '', _('Index')), 'modindex': ('py-modindex', '', _('Module Index')), 'search': ('search', '', _('Search Page'))}, 'anonlabels': {'genindex': ('genindex', ''), 'modindex': ('py-modindex', ''), 'search': ('search', '')}}
    dangling_warnings = {'term': 'term not in glossary: %(target)r', 'numref': 'undefined label: %(target)r', 'keyword': 'unknown keyword: %(target)r', 'doc': 'unknown document: %(target)r', 'option': 'unknown option: %(target)r'}
    enumerable_nodes: Dict[Type[Node], Tuple[str, Optional[Callable]]] = {nodes.figure: ('figure', None), nodes.table: ('table', None), nodes.container: ('code-block', None)}

    def __init__(self, env: 'BuildEnvironment') -> None:
        super().__init__(env)
        self.enumerable_nodes = copy(self.enumerable_nodes)
        for node, settings in env.app.registry.enumerable_nodes.items():
            self.enumerable_nodes[node] = settings

    def note_hyperlink_target(self, name: str, docname: str, node_id: str, title: str='') -> None:
        """Add a hyperlink target for cross reference.

        .. warning::

           This is only for internal use.  Please don't use this from your extension.
           ``document.note_explicit_target()`` or ``note_implicit_target()`` are recommended to
           add a hyperlink target to the document.

           This only adds a hyperlink target to the StandardDomain.  And this does not add a
           node_id to node.  Therefore, it is very fragile to calling this without
           understanding hyperlink target framework in both docutils and Sphinx.

        .. versionadded:: 3.0
        """
        if name in self.anonlabels and self.anonlabels[name] != (docname, node_id):
            logger.warning(__('duplicate label %s, other instance in %s'), name, self.env.doc2path(self.anonlabels[name][0]))
        self.anonlabels[name] = (docname, node_id)
        if title:
            self.labels[name] = (docname, node_id, title)

    @property
    def objects(self) -> Dict[Tuple[str, str], Tuple[str, str]]:
        return self.data.setdefault('objects', {})

    def note_object(self, objtype: str, name: str, labelid: str, location: Any=None) -> None:
        """Note a generic object for cross reference.

        .. versionadded:: 3.0
        """
        if (objtype, name) in self.objects:
            docname = self.objects[objtype, name][0]
            logger.warning(__('duplicate %s description of %s, other instance in %s'), objtype, name, docname, location=location)
        self.objects[objtype, name] = (self.env.docname, labelid)

    @property
    def _terms(self) -> Dict[str, Tuple[str, str]]:
        """.. note:: Will be removed soon. internal use only."""
        return self.data.setdefault('terms', {})

    def _note_term(self, term: str, labelid: str, location: Any=None) -> None:
        """Note a term for cross reference.

        .. note:: Will be removed soon. internal use only.
        """
        self.note_object('term', term, labelid, location)
        self._terms[term.lower()] = (self.env.docname, labelid)

    @property
    def progoptions(self) -> Dict[Tuple[str, str], Tuple[str, str]]:
        return self.data.setdefault('progoptions', {})

    @property
    def labels(self) -> Dict[str, Tuple[str, str, str]]:
        return self.data.setdefault('labels', {})

    @property
    def anonlabels(self) -> Dict[str, Tuple[str, str]]:
        return self.data.setdefault('anonlabels', {})

    def clear_doc(self, docname: str) -> None:
        key: Any = None
        for key, (fn, _l) in list(self.progoptions.items()):
            if fn == docname:
                del self.progoptions[key]
        for key, (fn, _l) in list(self.objects.items()):
            if fn == docname:
                del self.objects[key]
        for key, (fn, _l) in list(self._terms.items()):
            if fn == docname:
                del self._terms[key]
        for key, (fn, _l, _l) in list(self.labels.items()):
            if fn == docname:
                del self.labels[key]
        for key, (fn, _l) in list(self.anonlabels.items()):
            if fn == docname:
                del self.anonlabels[key]

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        for key, data in otherdata['progoptions'].items():
            if data[0] in docnames:
                self.progoptions[key] = data
        for key, data in otherdata['objects'].items():
            if data[0] in docnames:
                self.objects[key] = data
        for key, data in otherdata['terms'].items():
            if data[0] in docnames:
                self._terms[key] = data
        for key, data in otherdata['labels'].items():
            if data[0] in docnames:
                self.labels[key] = data
        for key, data in otherdata['anonlabels'].items():
            if data[0] in docnames:
                self.anonlabels[key] = data

    def process_doc(self, env: 'BuildEnvironment', docname: str, document: nodes.document) -> None:
        for name, explicit in document.nametypes.items():
            if not explicit:
                continue
            labelid = document.nameids[name]
            if labelid is None:
                continue
            node = document.ids[labelid]
            if isinstance(node, nodes.target) and 'refid' in node:
                node = document.ids.get(node['refid'])
                labelid = node['names'][0]
            if node.tagname == 'footnote' or 'refuri' in node or node.tagname.startswith('desc_'):
                continue
            if name in self.labels:
                logger.warning(__('duplicate label %s, other instance in %s'), name, env.doc2path(self.labels[name][0]), location=node)
            self.anonlabels[name] = (docname, labelid)
            if node.tagname == 'section':
                title = cast(nodes.title, node[0])
                sectname = clean_astext(title)
            elif node.tagname == 'rubric':
                sectname = clean_astext(node)
            elif self.is_enumerable_node(node):
                sectname = self.get_numfig_title(node)
                if not sectname:
                    continue
            else:
                if isinstance(node, (nodes.definition_list, nodes.field_list)) and node.children:
                    node = cast(nodes.Element, node.children[0])
                if isinstance(node, (nodes.field, nodes.definition_list_item)):
                    node = cast(nodes.Element, node.children[0])
                if isinstance(node, (nodes.term, nodes.field_name)):
                    sectname = clean_astext(node)
                else:
                    toctree = next(node.findall(addnodes.toctree), None)
                    if toctree and toctree.get('caption'):
                        sectname = toctree.get('caption')
                    else:
                        continue
            self.labels[name] = (docname, labelid, sectname)

    def add_program_option(self, program: str, name: str, docname: str, labelid: str) -> None:
        if (program, name) not in self.progoptions:
            self.progoptions[program, name] = (docname, labelid)

    def build_reference_node(self, fromdocname: str, builder: 'Builder', docname: str, labelid: str, sectname: str, rolename: str, **options: Any) -> Element:
        nodeclass = options.pop('nodeclass', nodes.reference)
        newnode = nodeclass('', '', internal=True, **options)
        innernode = nodes.inline(sectname, sectname)
        if innernode.get('classes') is not None:
            innernode['classes'].append('std')
            innernode['classes'].append('std-' + rolename)
        if docname == fromdocname:
            newnode['refid'] = labelid
        else:
            contnode = pending_xref('')
            contnode['refdocname'] = docname
            contnode['refsectname'] = sectname
            newnode['refuri'] = builder.get_relative_uri(fromdocname, docname)
            if labelid:
                newnode['refuri'] += '#' + labelid
        newnode.append(innernode)
        return newnode

    def resolve_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        if typ == 'ref':
            resolver = self._resolve_ref_xref
        elif typ == 'numref':
            resolver = self._resolve_numref_xref
        elif typ == 'keyword':
            resolver = self._resolve_keyword_xref
        elif typ == 'doc':
            resolver = self._resolve_doc_xref
        elif typ == 'option':
            resolver = self._resolve_option_xref
        elif typ == 'term':
            resolver = self._resolve_term_xref
        else:
            resolver = self._resolve_obj_xref
        return resolver(env, fromdocname, builder, typ, target, node, contnode)

    def _resolve_ref_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        if node['refexplicit']:
            docname, labelid = self.anonlabels.get(target, ('', ''))
            sectname = node.astext()
        else:
            docname, labelid, sectname = self.labels.get(target, ('', '', ''))
        if not docname:
            return None
        return self.build_reference_node(fromdocname, builder, docname, labelid, sectname, 'ref')

    def _resolve_numref_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        if target in self.labels:
            docname, labelid, figname = self.labels.get(target, ('', '', ''))
        else:
            docname, labelid = self.anonlabels.get(target, ('', ''))
            figname = None
        if not docname:
            return None
        target_node = env.get_doctree(docname).ids.get(labelid)
        figtype = self.get_enumerable_node_type(target_node)
        if figtype is None:
            return None
        if figtype != 'section' and env.config.numfig is False:
            logger.warning(__('numfig is disabled. :numref: is ignored.'), location=node)
            return contnode
        try:
            fignumber = self.get_fignumber(env, builder, figtype, docname, target_node)
            if fignumber is None:
                return contnode
        except ValueError:
            logger.warning(__('Failed to create a cross reference. Any number is not assigned: %s'), labelid, location=node)
            return contnode
        try:
            if node['refexplicit']:
                title = contnode.astext()
            else:
                title = env.config.numfig_format.get(figtype, '')
            if figname is None and '{name}' in title:
                logger.warning(__('the link has no caption: %s'), title, location=node)
                return contnode
            else:
                fignum = '.'.join(map(str, fignumber))
                if '{name}' in title or 'number' in title:
                    if figname:
                        newtitle = title.format(name=figname, number=fignum)
                    else:
                        newtitle = title.format(number=fignum)
                else:
                    newtitle = title % fignum
        except KeyError as exc:
            logger.warning(__('invalid numfig_format: %s (%r)'), title, exc, location=node)
            return contnode
        except TypeError:
            logger.warning(__('invalid numfig_format: %s'), title, location=node)
            return contnode
        return self.build_reference_node(fromdocname, builder, docname, labelid, newtitle, 'numref', nodeclass=addnodes.number_reference, title=title)

    def _resolve_keyword_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        docname, labelid, _ = self.labels.get(target, ('', '', ''))
        if not docname:
            return None
        return make_refnode(builder, fromdocname, docname, labelid, contnode)

    def _resolve_doc_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        refdoc = node.get('refdoc', fromdocname)
        docname = docname_join(refdoc, node['reftarget'])
        if docname not in env.all_docs:
            return None
        else:
            if node['refexplicit']:
                caption = node.astext()
            else:
                caption = clean_astext(env.titles[docname])
            innernode = nodes.inline(caption, caption, classes=['doc'])
            return make_refnode(builder, fromdocname, docname, None, innernode)

    def _resolve_option_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        progname = node.get('std:program')
        target = target.strip()
        docname, labelid = self.progoptions.get((progname, target), ('', ''))
        if not docname:
            for needle in {'=', '[=', ' '}:
                if needle in target:
                    stem, _, _ = target.partition(needle)
                    docname, labelid = self.progoptions.get((progname, stem), ('', ''))
                    if docname:
                        break
        if not docname:
            commands = []
            while ws_re.search(target):
                subcommand, target = ws_re.split(target, 1)
                commands.append(subcommand)
                progname = '-'.join(commands)
                docname, labelid = self.progoptions.get((progname, target), ('', ''))
                if docname:
                    break
            else:
                return None
        return make_refnode(builder, fromdocname, docname, labelid, contnode)

    def _resolve_term_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Element:
        result = self._resolve_obj_xref(env, fromdocname, builder, typ, target, node, contnode)
        if result:
            return result
        elif target.lower() in self._terms:
            docname, labelid = self._terms[target.lower()]
            return make_refnode(builder, fromdocname, docname, labelid, contnode)
        else:
            return None

    def _resolve_obj_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        objtypes = self.objtypes_for_role(typ) or []
        for objtype in objtypes:
            if (objtype, target) in self.objects:
                docname, labelid = self.objects[objtype, target]
                break
        else:
            docname, labelid = ('', '')
        if not docname:
            return None
        return make_refnode(builder, fromdocname, docname, labelid, contnode)

    def resolve_any_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', target: str, node: pending_xref, contnode: Element) -> List[Tuple[str, Element]]:
        results: List[Tuple[str, Element]] = []
        ltarget = target.lower()
        for role in ('ref', 'option'):
            res = self.resolve_xref(env, fromdocname, builder, role, ltarget if role == 'ref' else target, node, contnode)
            if res:
                results.append(('std:' + role, res))
        for objtype in self.object_types:
            key = (objtype, target)
            if objtype == 'term':
                key = (objtype, ltarget)
            if key in self.objects:
                docname, labelid = self.objects[key]
                results.append(('std:' + self.role_for_objtype(objtype), make_refnode(builder, fromdocname, docname, labelid, contnode)))
        return results

    def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
        for doc in self.env.all_docs:
            yield (doc, clean_astext(self.env.titles[doc]), 'doc', doc, '', -1)
        for (prog, option), info in self.progoptions.items():
            if prog:
                fullname = '.'.join([prog, option])
                yield (fullname, fullname, 'cmdoption', info[0], info[1], 1)
            else:
                yield (option, option, 'cmdoption', info[0], info[1], 1)
        for (type, name), info in self.objects.items():
            yield (name, name, type, info[0], info[1], self.object_types[type].attrs['searchprio'])
        for name, (docname, labelid, sectionname) in self.labels.items():
            yield (name, sectionname, 'label', docname, labelid, -1)
        non_anon_labels = set(self.labels)
        for name, (docname, labelid) in self.anonlabels.items():
            if name not in non_anon_labels:
                yield (name, name, 'label', docname, labelid, -1)

    def get_type_name(self, type: ObjType, primary: bool=False) -> str:
        return type.lname

    def is_enumerable_node(self, node: Node) -> bool:
        return node.__class__ in self.enumerable_nodes

    def get_numfig_title(self, node: Node) -> Optional[str]:
        """Get the title of enumerable nodes to refer them using its title"""
        if self.is_enumerable_node(node):
            elem = cast(Element, node)
            _, title_getter = self.enumerable_nodes.get(elem.__class__, (None, None))
            if title_getter:
                return title_getter(elem)
            else:
                for subnode in elem:
                    if isinstance(subnode, (nodes.caption, nodes.title)):
                        return clean_astext(subnode)
        return None

    def get_enumerable_node_type(self, node: Node) -> Optional[str]:
        """Get type of enumerable nodes."""

        def has_child(node: Element, cls: Type) -> bool:
            return any((isinstance(child, cls) for child in node))
        if isinstance(node, nodes.section):
            return 'section'
        elif isinstance(node, nodes.container) and 'literal_block' in node and has_child(node, nodes.literal_block):
            return 'code-block'
        else:
            figtype, _ = self.enumerable_nodes.get(node.__class__, (None, None))
            return figtype

    def get_fignumber(self, env: 'BuildEnvironment', builder: 'Builder', figtype: str, docname: str, target_node: Element) -> Tuple[int, ...]:
        if figtype == 'section':
            if builder.name == 'latex':
                return ()
            elif docname not in env.toc_secnumbers:
                raise ValueError
            else:
                anchorname = '#' + target_node['ids'][0]
                if anchorname not in env.toc_secnumbers[docname]:
                    return env.toc_secnumbers[docname].get('')
                else:
                    return env.toc_secnumbers[docname].get(anchorname)
        else:
            try:
                figure_id = target_node['ids'][0]
                return env.toc_fignumbers[docname][figtype][figure_id]
            except (KeyError, IndexError) as exc:
                raise ValueError from exc

    def get_full_qualified_name(self, node: Element) -> Optional[str]:
        if node.get('reftype') == 'option':
            progname = node.get('std:program')
            command = ws_re.split(node.get('reftarget'))
            if progname:
                command.insert(0, progname)
            option = command.pop()
            if command:
                return '.'.join(['-'.join(command), option])
            else:
                return None
        else:
            return None