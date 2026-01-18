import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
import docutils.parsers.rst.directives
import docutils.parsers.rst.roles
import docutils.parsers.rst.states
from docutils import nodes, utils
from docutils.nodes import Element, Node, TextElement, system_message
from sphinx import addnodes
from sphinx.locale import _, __
from sphinx.util import ws_re
from sphinx.util.docutils import ReferenceRole, SphinxRole
from sphinx.util.typing import RoleFunction
class XRefRole(ReferenceRole):
    """
    A generic cross-referencing role.  To create a callable that can be used as
    a role function, create an instance of this class.

    The general features of this role are:

    * Automatic creation of a reference and a content node.
    * Optional separation of title and target with `title <target>`.
    * The implementation is a class rather than a function to make
      customization easier.

    Customization can be done in two ways:

    * Supplying constructor parameters:
      * `fix_parens` to normalize parentheses (strip from target, and add to
        title if configured)
      * `lowercase` to lowercase the target
      * `nodeclass` and `innernodeclass` select the node classes for
        the reference and the content node

    * Subclassing and overwriting `process_link()` and/or `result_nodes()`.
    """
    nodeclass: Type[Element] = addnodes.pending_xref
    innernodeclass: Type[TextElement] = nodes.literal

    def __init__(self, fix_parens: bool=False, lowercase: bool=False, nodeclass: Optional[Type[Element]]=None, innernodeclass: Optional[Type[TextElement]]=None, warn_dangling: bool=False) -> None:
        self.fix_parens = fix_parens
        self.lowercase = lowercase
        self.warn_dangling = warn_dangling
        if nodeclass is not None:
            self.nodeclass = nodeclass
        if innernodeclass is not None:
            self.innernodeclass = innernodeclass
        super().__init__()

    def update_title_and_target(self, title: str, target: str) -> Tuple[str, str]:
        if not self.has_explicit_title:
            if title.endswith('()'):
                title = title[:-2]
            if self.config.add_function_parentheses:
                title += '()'
        if target.endswith('()'):
            target = target[:-2]
        return (title, target)

    def run(self) -> Tuple[List[Node], List[system_message]]:
        if ':' not in self.name:
            self.refdomain, self.reftype = ('', self.name)
            self.classes = ['xref', self.reftype]
        else:
            self.refdomain, self.reftype = self.name.split(':', 1)
            self.classes = ['xref', self.refdomain, '%s-%s' % (self.refdomain, self.reftype)]
        if self.disabled:
            return self.create_non_xref_node()
        else:
            return self.create_xref_node()

    def create_non_xref_node(self) -> Tuple[List[Node], List[system_message]]:
        text = utils.unescape(self.text[1:])
        if self.fix_parens:
            self.has_explicit_title = False
            text, target = self.update_title_and_target(text, '')
        node = self.innernodeclass(self.rawtext, text, classes=self.classes)
        return self.result_nodes(self.inliner.document, self.env, node, is_ref=False)

    def create_xref_node(self) -> Tuple[List[Node], List[system_message]]:
        target = self.target
        title = self.title
        if self.lowercase:
            target = target.lower()
        if self.fix_parens:
            title, target = self.update_title_and_target(title, target)
        options = {'refdoc': self.env.docname, 'refdomain': self.refdomain, 'reftype': self.reftype, 'refexplicit': self.has_explicit_title, 'refwarn': self.warn_dangling}
        refnode = self.nodeclass(self.rawtext, **options)
        self.set_source_info(refnode)
        title, target = self.process_link(self.env, refnode, self.has_explicit_title, title, target)
        refnode['reftarget'] = target
        refnode += self.innernodeclass(self.rawtext, title, classes=self.classes)
        return self.result_nodes(self.inliner.document, self.env, refnode, is_ref=True)

    def process_link(self, env: 'BuildEnvironment', refnode: Element, has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        """Called after parsing title and target text, and creating the
        reference node (given in *refnode*).  This method can alter the
        reference node and must return a new (or the same) ``(title, target)``
        tuple.
        """
        return (title, ws_re.sub(' ', target))

    def result_nodes(self, document: nodes.document, env: 'BuildEnvironment', node: Element, is_ref: bool) -> Tuple[List[Node], List[system_message]]:
        """Called before returning the finished nodes.  *node* is the reference
        node if one was created (*is_ref* is then true), else the content node.
        This method can add other nodes and must return a ``(nodes, messages)``
        tuple (the usual return value of a role function).
        """
        return ([node], [])