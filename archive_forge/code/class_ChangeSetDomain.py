from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.domains import Domain
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
class ChangeSetDomain(Domain):
    """Domain for changesets."""
    name = 'changeset'
    label = 'changeset'
    initial_data: Dict = {'changes': {}}

    @property
    def changesets(self) -> Dict[str, List[ChangeSet]]:
        return self.data.setdefault('changes', {})

    def note_changeset(self, node: addnodes.versionmodified) -> None:
        version = node['version']
        module = self.env.ref_context.get('py:module')
        objname = self.env.temp_data.get('object')
        changeset = ChangeSet(node['type'], self.env.docname, node.line, module, objname, node.astext())
        self.changesets.setdefault(version, []).append(changeset)

    def clear_doc(self, docname: str) -> None:
        for changes in self.changesets.values():
            for changeset in changes[:]:
                if changeset.docname == docname:
                    changes.remove(changeset)

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        for version, otherchanges in otherdata['changes'].items():
            changes = self.changesets.setdefault(version, [])
            for changeset in otherchanges:
                if changeset.docname in docnames:
                    changes.append(changeset)

    def process_doc(self, env: 'BuildEnvironment', docname: str, document: nodes.document) -> None:
        pass

    def get_changesets_for(self, version: str) -> List[ChangeSet]:
        return self.changesets.get(version, [])