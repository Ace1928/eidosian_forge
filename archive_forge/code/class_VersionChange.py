from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.domains import Domain
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
class VersionChange(SphinxDirective):
    """
    Directive to describe a change/addition/deprecation in a specific version.
    """
    has_content = True
    required_arguments = 1
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec: OptionSpec = {}

    def run(self) -> List[Node]:
        node = addnodes.versionmodified()
        node.document = self.state.document
        self.set_source_info(node)
        node['type'] = self.name
        node['version'] = self.arguments[0]
        text = versionlabels[self.name] % self.arguments[0]
        if len(self.arguments) == 2:
            inodes, messages = self.state.inline_text(self.arguments[1], self.lineno + 1)
            para = nodes.paragraph(self.arguments[1], '', *inodes, translatable=False)
            self.set_source_info(para)
            node.append(para)
        else:
            messages = []
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        classes = ['versionmodified', versionlabel_classes[self.name]]
        if len(node) > 0 and isinstance(node[0], nodes.paragraph):
            if node[0].rawsource:
                content = nodes.inline(node[0].rawsource, translatable=True)
                content.source = node[0].source
                content.line = node[0].line
                content += node[0].children
                node[0].replace_self(nodes.paragraph('', '', content, translatable=False))
            para = node[0]
            para.insert(0, nodes.inline('', '%s: ' % text, classes=classes))
        elif len(node) > 0:
            para = nodes.paragraph('', '', nodes.inline('', '%s: ' % text, classes=classes), translatable=False)
            node.insert(0, para)
        else:
            para = nodes.paragraph('', '', nodes.inline('', '%s.' % text, classes=classes), translatable=False)
            node.append(para)
        domain = cast(ChangeSetDomain, self.env.get_domain('changeset'))
        domain.note_changeset(node)
        ret: List[Node] = [node]
        ret += messages
        return ret