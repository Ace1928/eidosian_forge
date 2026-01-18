from os import path
from typing import Any, Dict, List, Optional, Tuple, Union
from docutils import nodes
from docutils.nodes import Node
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.toctree import TocTree
from sphinx.locale import __
from sphinx.util import logging, progress_message
from sphinx.util.console import darkgreen  # type: ignore
from sphinx.util.nodes import inline_all_toctrees
class SingleFileHTMLBuilder(StandaloneHTMLBuilder):
    """
    A StandaloneHTMLBuilder subclass that puts the whole document tree on one
    HTML page.
    """
    name = 'singlehtml'
    epilog = __('The HTML page is in %(outdir)s.')
    copysource = False

    def get_outdated_docs(self) -> Union[str, List[str]]:
        return 'all documents'

    def get_target_uri(self, docname: str, typ: Optional[str]=None) -> str:
        if docname in self.env.all_docs:
            return self.config.root_doc + self.out_suffix + '#document-' + docname
        else:
            return docname + self.out_suffix

    def get_relative_uri(self, from_: str, to: str, typ: Optional[str]=None) -> str:
        return self.get_target_uri(to, typ)

    def fix_refuris(self, tree: Node) -> None:
        fname = self.config.root_doc + self.out_suffix
        for refnode in tree.findall(nodes.reference):
            if 'refuri' not in refnode:
                continue
            refuri = refnode['refuri']
            hashindex = refuri.find('#')
            if hashindex < 0:
                continue
            hashindex = refuri.find('#', hashindex + 1)
            if hashindex >= 0:
                refnode['refuri'] = fname + refuri[hashindex:]

    def _get_local_toctree(self, docname: str, collapse: bool=True, **kwargs: Any) -> str:
        if 'includehidden' not in kwargs:
            kwargs['includehidden'] = False
        toctree = TocTree(self.env).get_toctree_for(docname, self, collapse, **kwargs)
        if toctree is not None:
            self.fix_refuris(toctree)
        return self.render_partial(toctree)['fragment']

    def assemble_doctree(self) -> nodes.document:
        master = self.config.root_doc
        tree = self.env.get_doctree(master)
        tree = inline_all_toctrees(self, set(), master, tree, darkgreen, [master])
        tree['docname'] = master
        self.env.resolve_references(tree, master, self)
        self.fix_refuris(tree)
        return tree

    def assemble_toc_secnumbers(self) -> Dict[str, Dict[str, Tuple[int, ...]]]:
        new_secnumbers: Dict[str, Tuple[int, ...]] = {}
        for docname, secnums in self.env.toc_secnumbers.items():
            for id, secnum in secnums.items():
                alias = '%s/%s' % (docname, id)
                new_secnumbers[alias] = secnum
        return {self.config.root_doc: new_secnumbers}

    def assemble_toc_fignumbers(self) -> Dict[str, Dict[str, Dict[str, Tuple[int, ...]]]]:
        new_fignumbers: Dict[str, Dict[str, Tuple[int, ...]]] = {}
        for docname, fignumlist in self.env.toc_fignumbers.items():
            for figtype, fignums in fignumlist.items():
                alias = '%s/%s' % (docname, figtype)
                new_fignumbers.setdefault(alias, {})
                for id, fignum in fignums.items():
                    new_fignumbers[alias][id] = fignum
        return {self.config.root_doc: new_fignumbers}

    def get_doc_context(self, docname: str, body: str, metatags: str) -> Dict[str, Any]:
        toctree = TocTree(self.env).get_toctree_for(self.config.root_doc, self, False)
        if toctree:
            self.fix_refuris(toctree)
            toc = self.render_partial(toctree)['fragment']
            display_toc = True
        else:
            toc = ''
            display_toc = False
        return {'parents': [], 'prev': None, 'next': None, 'docstitle': None, 'title': self.config.html_title, 'meta': None, 'body': body, 'metatags': metatags, 'rellinks': [], 'sourcename': '', 'toc': toc, 'display_toc': display_toc}

    def write(self, *ignored: Any) -> None:
        docnames = self.env.all_docs
        with progress_message(__('preparing documents')):
            self.prepare_writing(docnames)
        with progress_message(__('assembling single document')):
            doctree = self.assemble_doctree()
            self.env.toc_secnumbers = self.assemble_toc_secnumbers()
            self.env.toc_fignumbers = self.assemble_toc_fignumbers()
        with progress_message(__('writing')):
            self.write_doc_serialized(self.config.root_doc, doctree)
            self.write_doc(self.config.root_doc, doctree)

    def finish(self) -> None:
        self.write_additional_files()
        self.copy_image_files()
        self.copy_download_files()
        self.copy_static_files()
        self.copy_extra_files()
        self.write_buildinfo()
        self.dump_inventory()

    @progress_message(__('writing additional files'))
    def write_additional_files(self) -> None:
        for pagename, template in self.config.html_additional_pages.items():
            logger.info(' ' + pagename, nonl=True)
            self.handle_page(pagename, {}, template)
        if self.config.html_use_opensearch:
            logger.info(' opensearch', nonl=True)
            fn = path.join(self.outdir, '_static', 'opensearch.xml')
            self.handle_page('opensearch', {}, 'opensearch.xml', outfilename=fn)