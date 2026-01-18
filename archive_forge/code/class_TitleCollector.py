from typing import Any, Dict, Set
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.transforms import SphinxContentsFilter
class TitleCollector(EnvironmentCollector):
    """title collector for sphinx.environment."""

    def clear_doc(self, app: Sphinx, env: BuildEnvironment, docname: str) -> None:
        env.titles.pop(docname, None)
        env.longtitles.pop(docname, None)

    def merge_other(self, app: Sphinx, env: BuildEnvironment, docnames: Set[str], other: BuildEnvironment) -> None:
        for docname in docnames:
            env.titles[docname] = other.titles[docname]
            env.longtitles[docname] = other.longtitles[docname]

    def process_doc(self, app: Sphinx, doctree: nodes.document) -> None:
        """Add a title node to the document (just copy the first section title),
        and store that title in the environment.
        """
        titlenode = nodes.title()
        longtitlenode = titlenode
        if 'title' in doctree:
            longtitlenode = nodes.title()
            longtitlenode += nodes.Text(doctree['title'])
        for node in doctree.findall(nodes.section):
            visitor = SphinxContentsFilter(doctree)
            node[0].walkabout(visitor)
            titlenode += visitor.get_entry_text()
            break
        else:
            titlenode += nodes.Text(doctree.get('title', '<no title>'))
        app.env.titles[app.env.docname] = titlenode
        app.env.longtitles[app.env.docname] = longtitlenode