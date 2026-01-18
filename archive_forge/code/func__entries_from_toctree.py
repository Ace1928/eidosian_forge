from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging, url_re
from sphinx.util.matching import Matcher
from sphinx.util.nodes import clean_astext, process_only_nodes
def _entries_from_toctree(toctreenode: addnodes.toctree, parents: List[str], separate: bool=False, subtree: bool=False) -> List[Element]:
    """Return TOC entries for a toctree node."""
    refs = [(e[0], e[1]) for e in toctreenode['entries']]
    entries: List[Element] = []
    for title, ref in refs:
        try:
            refdoc = None
            if url_re.match(ref):
                if title is None:
                    title = ref
                reference = nodes.reference('', '', *[nodes.Text(title)], internal=False, refuri=ref, anchorname='')
                para = addnodes.compact_paragraph('', '', reference)
                item = nodes.list_item('', para)
                toc = nodes.bullet_list('', item)
            elif ref == 'self':
                ref = toctreenode['parent']
                if not title:
                    title = clean_astext(self.env.titles[ref])
                reference = nodes.reference('', '', *[nodes.Text(title)], internal=True, refuri=ref, anchorname='')
                para = addnodes.compact_paragraph('', '', reference)
                item = nodes.list_item('', para)
                toc = nodes.bullet_list('', item)
            elif ref in generated_docnames:
                docname, _, sectionname = generated_docnames[ref]
                if not title:
                    title = sectionname
                reference = nodes.reference('', title, internal=True, refuri=docname, anchorname='')
                para = addnodes.compact_paragraph('', '', reference)
                item = nodes.list_item('', para)
                toc = nodes.bullet_list('', item)
            else:
                if ref in parents:
                    logger.warning(__('circular toctree references detected, ignoring: %s <- %s'), ref, ' <- '.join(parents), location=ref, type='toc', subtype='circular')
                    continue
                refdoc = ref
                toc = self.env.tocs[ref].deepcopy()
                maxdepth = self.env.metadata[ref].get('tocdepth', 0)
                if ref not in toctree_ancestors or (prune and maxdepth > 0):
                    self._toctree_prune(toc, 2, maxdepth, collapse)
                process_only_nodes(toc, builder.tags)
                if title and toc.children and (len(toc.children) == 1):
                    child = toc.children[0]
                    for refnode in child.findall(nodes.reference):
                        if refnode['refuri'] == ref and (not refnode['anchorname']):
                            refnode.children = [nodes.Text(title)]
            if not toc.children:
                logger.warning(__("toctree contains reference to document %r that doesn't have a title: no link will be generated"), ref, location=toctreenode)
        except KeyError:
            if excluded(self.env.doc2path(ref, False)):
                message = __('toctree contains reference to excluded document %r')
            elif not included(self.env.doc2path(ref, False)):
                message = __('toctree contains reference to non-included document %r')
            else:
                message = __('toctree contains reference to nonexisting document %r')
            logger.warning(message, ref, location=toctreenode)
        else:
            if titles_only:
                children = cast(Iterable[nodes.Element], toc)
                for toplevel in children:
                    if len(toplevel) > 1:
                        subtrees = list(toplevel.findall(addnodes.toctree))
                        if subtrees:
                            toplevel[1][:] = subtrees
                        else:
                            toplevel.pop(1)
            for subtocnode in list(toc.findall(addnodes.toctree)):
                if not (subtocnode.get('hidden', False) and (not includehidden)):
                    i = subtocnode.parent.index(subtocnode) + 1
                    for entry in _entries_from_toctree(subtocnode, [refdoc] + parents, subtree=True):
                        subtocnode.parent.insert(i, entry)
                        i += 1
                    subtocnode.parent.remove(subtocnode)
            if separate:
                entries.append(toc)
            else:
                children = cast(Iterable[nodes.Element], toc)
                entries.extend(children)
    if not subtree and (not separate):
        ret = nodes.bullet_list()
        ret += entries
        return [ret]
    return entries