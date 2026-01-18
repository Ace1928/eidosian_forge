from typing import Any, Dict, List, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.transforms.references import Substitutions
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.latex.nodes import (captioned_literal_block, footnotemark, footnotetext,
from sphinx.domains.citation import CitationDomain
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.nodes import NodeMatcher
class ShowUrlsTransform(SphinxPostTransform):
    """Expand references to inline text or footnotes.

    For more information, see :confval:`latex_show_urls`.

    .. note:: This transform is used for integrated doctree
    """
    default_priority = 400
    formats = ('latex',)
    expanded = False

    def run(self, **kwargs: Any) -> None:
        try:
            settings: Any = self.document.settings
            id_prefix = settings.id_prefix
            settings.id_prefix = 'show_urls'
            self.expand_show_urls()
            if self.expanded:
                self.renumber_footnotes()
        finally:
            settings.id_prefix = id_prefix

    def expand_show_urls(self) -> None:
        show_urls = self.config.latex_show_urls
        if show_urls is False or show_urls == 'no':
            return
        for node in list(self.document.findall(nodes.reference)):
            uri = node.get('refuri', '')
            if uri.startswith(URI_SCHEMES):
                if uri.startswith('mailto:'):
                    uri = uri[7:]
                if node.astext() != uri:
                    index = node.parent.index(node)
                    docname = self.get_docname_for_node(node)
                    if show_urls == 'footnote':
                        fn, fnref = self.create_footnote(uri, docname)
                        node.parent.insert(index + 1, fn)
                        node.parent.insert(index + 2, fnref)
                        self.expanded = True
                    else:
                        textnode = nodes.Text(' (%s)' % uri)
                        node.parent.insert(index + 1, textnode)

    def get_docname_for_node(self, node: Node) -> str:
        while node:
            if isinstance(node, nodes.document):
                return self.env.path2doc(node['source']) or ''
            elif isinstance(node, addnodes.start_of_file):
                return node['docname']
            else:
                node = node.parent
        try:
            source = node['source']
        except TypeError:
            raise ValueError('Failed to get a docname!') from None
        raise ValueError(f'Failed to get a docname for source {source!r}!')

    def create_footnote(self, uri: str, docname: str) -> Tuple[nodes.footnote, nodes.footnote_reference]:
        reference = nodes.reference('', nodes.Text(uri), refuri=uri, nolinkurl=True)
        footnote = nodes.footnote(uri, auto=1, docname=docname)
        footnote['names'].append('#')
        footnote += nodes.label('', '#')
        footnote += nodes.paragraph('', '', reference)
        self.document.note_autofootnote(footnote)
        footnote_ref = nodes.footnote_reference('[#]_', auto=1, refid=footnote['ids'][0], docname=docname)
        footnote_ref += nodes.Text('#')
        self.document.note_autofootnote_ref(footnote_ref)
        footnote.add_backref(footnote_ref['ids'][0])
        return (footnote, footnote_ref)

    def renumber_footnotes(self) -> None:
        collector = FootnoteCollector(self.document)
        self.document.walkabout(collector)
        num = 0
        for footnote in collector.auto_footnotes:
            while True:
                num += 1
                if str(num) not in collector.used_footnote_numbers:
                    break
            old_label = cast(nodes.label, footnote[0])
            old_label.replace_self(nodes.label('', str(num)))
            if old_label in footnote['names']:
                footnote['names'].remove(old_label.astext())
            footnote['names'].append(str(num))
            docname = footnote['docname']
            for ref in collector.footnote_refs:
                if docname == ref['docname'] and footnote['ids'][0] == ref['refid']:
                    ref.remove(ref[0])
                    ref += nodes.Text(str(num))