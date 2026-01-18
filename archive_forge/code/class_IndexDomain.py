from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple
from docutils import nodes
from docutils.nodes import Node, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.util import logging, split_index_msg
from sphinx.util.docutils import ReferenceRole, SphinxDirective
from sphinx.util.nodes import process_index_entry
from sphinx.util.typing import OptionSpec
class IndexDomain(Domain):
    """Mathematics domain."""
    name = 'index'
    label = 'index'

    @property
    def entries(self) -> Dict[str, List[Tuple[str, str, str, str, str]]]:
        return self.data.setdefault('entries', {})

    def clear_doc(self, docname: str) -> None:
        self.entries.pop(docname, None)

    def merge_domaindata(self, docnames: Iterable[str], otherdata: Dict) -> None:
        for docname in docnames:
            self.entries[docname] = otherdata['entries'][docname]

    def process_doc(self, env: BuildEnvironment, docname: str, document: Node) -> None:
        """Process a document after it is read by the environment."""
        entries = self.entries.setdefault(env.docname, [])
        for node in list(document.findall(addnodes.index)):
            try:
                for entry in node['entries']:
                    split_index_msg(entry[0], entry[1])
            except ValueError as exc:
                logger.warning(str(exc), location=node)
                node.parent.remove(node)
            else:
                for entry in node['entries']:
                    entries.append(entry)