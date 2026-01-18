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
class RFC(ReferenceRole):

    def run(self) -> Tuple[List[Node], List[system_message]]:
        target_id = 'index-%s' % self.env.new_serialno('index')
        entries = [('single', 'RFC; RFC %s' % self.target, target_id, '', None)]
        index = addnodes.index(entries=entries)
        target = nodes.target('', '', ids=[target_id])
        self.inliner.document.note_explicit_target(target)
        try:
            refuri = self.build_uri()
            reference = nodes.reference('', '', internal=False, refuri=refuri, classes=['rfc'])
            if self.has_explicit_title:
                reference += nodes.strong(self.title, self.title)
            else:
                title = 'RFC ' + self.title
                reference += nodes.strong(title, title)
        except ValueError:
            msg = self.inliner.reporter.error(__('invalid RFC number %s') % self.target, line=self.lineno)
            prb = self.inliner.problematic(self.rawtext, self.rawtext, msg)
            return ([prb], [msg])
        return ([index, target, reference], [])

    def build_uri(self) -> str:
        base_url = self.inliner.document.settings.rfc_base_url
        ret = self.target.split('#', 1)
        if len(ret) == 2:
            return base_url + self.inliner.rfc_url % int(ret[0]) + '#' + ret[1]
        else:
            return base_url + self.inliner.rfc_url % int(ret[0])