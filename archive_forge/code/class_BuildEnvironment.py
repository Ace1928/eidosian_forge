import os
import pickle
import warnings
from collections import defaultdict
from copy import copy
from datetime import datetime
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, Optional,
import docutils
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.domains import Domain
from sphinx.environment.adapters.toctree import TocTree
from sphinx.errors import BuildEnvironmentError, DocumentError, ExtensionError, SphinxError
from sphinx.events import EventManager
from sphinx.locale import __
from sphinx.project import Project
from sphinx.transforms import SphinxTransformer
from sphinx.util import DownloadFiles, FilenameUniqDict, logging
from sphinx.util.docutils import LoggingReporter
from sphinx.util.i18n import CatalogRepository, docname_to_domain
from sphinx.util.nodes import is_translatable
from sphinx.util.osutil import canon_path, os_path
class BuildEnvironment:
    """
    The environment in which the ReST files are translated.
    Stores an inventory of cross-file targets and provides doctree
    transformations to resolve links to them.
    """
    domains: _DomainsType

    def __init__(self, app: Optional['Sphinx']=None):
        self.app: Sphinx = None
        self.doctreedir: str = None
        self.srcdir: str = None
        self.config: Config = None
        self.config_status: int = None
        self.config_status_extra: str = None
        self.events: EventManager = None
        self.project: Project = None
        self.version: Dict[str, str] = None
        self.versioning_condition: Union[bool, Callable] = None
        self.versioning_compare: bool = None
        self.domains = _DomainsType()
        self.settings = default_settings.copy()
        self.settings['env'] = self
        self.all_docs: Dict[str, float] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.included: Dict[str, Set[str]] = defaultdict(set)
        self.reread_always: Set[str] = set()
        self.metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.titles: Dict[str, nodes.title] = {}
        self.longtitles: Dict[str, nodes.title] = {}
        self.tocs: Dict[str, nodes.bullet_list] = {}
        self.toc_num_entries: Dict[str, int] = {}
        self.toc_secnumbers: Dict[str, Dict[str, Tuple[int, ...]]] = {}
        self.toc_fignumbers: Dict[str, Dict[str, Dict[str, Tuple[int, ...]]]] = {}
        self.toctree_includes: Dict[str, List[str]] = {}
        self.files_to_rebuild: Dict[str, Set[str]] = {}
        self.glob_toctrees: Set[str] = set()
        self.numbered_toctrees: Set[str] = set()
        self.domaindata: Dict[str, Dict] = {}
        self.images: FilenameUniqDict = FilenameUniqDict()
        self.dlfiles: DownloadFiles = DownloadFiles()
        self.original_image_uri: Dict[str, str] = {}
        self.temp_data: Dict[str, Any] = {}
        self.ref_context: Dict[str, Any] = {}
        if app:
            self.setup(app)
        else:
            warnings.warn("The 'app' argument for BuildEnvironment() becomes required now.", RemovedInSphinx60Warning, stacklevel=2)

    def __getstate__(self) -> Dict:
        """Obtains serializable data for pickling."""
        __dict__ = self.__dict__.copy()
        __dict__.update(app=None, domains={}, events=None)
        return __dict__

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)

    def setup(self, app: 'Sphinx') -> None:
        """Set up BuildEnvironment object."""
        if self.version and self.version != app.registry.get_envversion(app):
            raise BuildEnvironmentError(__('build environment version not current'))
        elif self.srcdir and self.srcdir != app.srcdir:
            raise BuildEnvironmentError(__('source directory has changed'))
        if self.project:
            app.project.restore(self.project)
        self.app = app
        self.doctreedir = app.doctreedir
        self.events = app.events
        self.srcdir = app.srcdir
        self.project = app.project
        self.version = app.registry.get_envversion(app)
        self.domains = _DomainsType()
        for domain in app.registry.create_domains(self):
            self.domains[domain.name] = domain
        for domain in self.domains.values():
            domain.setup()
        self._update_config(app.config)
        self._update_settings(app.config)

    def _update_config(self, config: Config) -> None:
        """Update configurations by new one."""
        self.config_status = CONFIG_OK
        self.config_status_extra = ''
        if self.config is None:
            self.config_status = CONFIG_NEW
        elif self.config.extensions != config.extensions:
            self.config_status = CONFIG_EXTENSIONS_CHANGED
            extensions = sorted(set(self.config.extensions) ^ set(config.extensions))
            if len(extensions) == 1:
                extension = extensions[0]
            else:
                extension = '%d' % (len(extensions),)
            self.config_status_extra = ' (%r)' % (extension,)
        else:
            for item in config.filter('env'):
                if self.config[item.name] != item.value:
                    self.config_status = CONFIG_CHANGED
                    self.config_status_extra = ' (%r)' % (item.name,)
                    break
        self.config = config

    def _update_settings(self, config: Config) -> None:
        """Update settings by new config."""
        self.settings['input_encoding'] = config.source_encoding
        self.settings['trim_footnote_reference_space'] = config.trim_footnote_reference_space
        self.settings['language_code'] = config.language
        self.settings.setdefault('smart_quotes', True)

    def set_versioning_method(self, method: Union[str, Callable], compare: bool) -> None:
        """This sets the doctree versioning method for this environment.

        Versioning methods are a builder property; only builders with the same
        versioning method can share the same doctree directory.  Therefore, we
        raise an exception if the user tries to use an environment with an
        incompatible versioning method.
        """
        condition: Union[bool, Callable]
        if callable(method):
            condition = method
        else:
            if method not in versioning_conditions:
                raise ValueError('invalid versioning method: %r' % method)
            condition = versioning_conditions[method]
        if self.versioning_condition not in (None, condition):
            raise SphinxError(__('This environment is incompatible with the selected builder, please choose another doctree directory.'))
        self.versioning_condition = condition
        self.versioning_compare = compare

    def clear_doc(self, docname: str) -> None:
        """Remove all traces of a source file in the inventory."""
        if docname in self.all_docs:
            self.all_docs.pop(docname, None)
            self.included.pop(docname, None)
            self.reread_always.discard(docname)
        for domain in self.domains.values():
            domain.clear_doc(docname)

    def merge_info_from(self, docnames: List[str], other: 'BuildEnvironment', app: 'Sphinx') -> None:
        """Merge global information gathered about *docnames* while reading them
        from the *other* environment.

        This possibly comes from a parallel build process.
        """
        docnames = set(docnames)
        for docname in docnames:
            self.all_docs[docname] = other.all_docs[docname]
            self.included[docname] = other.included[docname]
            if docname in other.reread_always:
                self.reread_always.add(docname)
        for domainname, domain in self.domains.items():
            domain.merge_domaindata(docnames, other.domaindata[domainname])
        self.events.emit('env-merge-info', self, docnames, other)

    def path2doc(self, filename: str) -> Optional[str]:
        """Return the docname for the filename if the file is document.

        *filename* should be absolute or relative to the source directory.
        """
        return self.project.path2doc(filename)

    def doc2path(self, docname: str, base: bool=True) -> str:
        """Return the filename for the document name.

        If *base* is True, return absolute path under self.srcdir.
        If *base* is False, return relative path to self.srcdir.
        """
        return self.project.doc2path(docname, base)

    def relfn2path(self, filename: str, docname: Optional[str]=None) -> Tuple[str, str]:
        """Return paths to a file referenced from a document, relative to
        documentation root and absolute.

        In the input "filename", absolute filenames are taken as relative to the
        source dir, while relative filenames are relative to the dir of the
        containing document.
        """
        filename = os_path(filename)
        if filename.startswith('/') or filename.startswith(os.sep):
            rel_fn = filename[1:]
        else:
            docdir = path.dirname(self.doc2path(docname or self.docname, base=False))
            rel_fn = path.join(docdir, filename)
        return (canon_path(path.normpath(rel_fn)), path.normpath(path.join(self.srcdir, rel_fn)))

    @property
    def found_docs(self) -> Set[str]:
        """contains all existing docnames."""
        return self.project.docnames

    def find_files(self, config: Config, builder: 'Builder') -> None:
        """Find all source files in the source dir and put them in
        self.found_docs.
        """
        try:
            exclude_paths = self.config.exclude_patterns + self.config.templates_path + builder.get_asset_paths()
            self.project.discover(exclude_paths, self.config.include_patterns)
            if builder.use_message_catalog:
                repo = CatalogRepository(self.srcdir, self.config.locale_dirs, self.config.language, self.config.source_encoding)
                mo_paths = {c.domain: c.mo_path for c in repo.catalogs}
                for docname in self.found_docs:
                    domain = docname_to_domain(docname, self.config.gettext_compact)
                    if domain in mo_paths:
                        self.dependencies[docname].add(mo_paths[domain])
        except OSError as exc:
            raise DocumentError(__('Failed to scan documents in %s: %r') % (self.srcdir, exc)) from exc

    def get_outdated_files(self, config_changed: bool) -> Tuple[Set[str], Set[str], Set[str]]:
        """Return (added, changed, removed) sets."""
        removed = set(self.all_docs) - self.found_docs
        added: Set[str] = set()
        changed: Set[str] = set()
        if config_changed:
            added = self.found_docs
        else:
            for docname in self.found_docs:
                if docname not in self.all_docs:
                    logger.debug('[build target] added %r', docname)
                    added.add(docname)
                    continue
                filename = path.join(self.doctreedir, docname + '.doctree')
                if not path.isfile(filename):
                    logger.debug('[build target] changed %r', docname)
                    changed.add(docname)
                    continue
                if docname in self.reread_always:
                    logger.debug('[build target] changed %r', docname)
                    changed.add(docname)
                    continue
                mtime = self.all_docs[docname]
                newmtime = path.getmtime(self.doc2path(docname))
                if newmtime > mtime:
                    logger.debug('[build target] outdated %r: %s -> %s', docname, datetime.utcfromtimestamp(mtime), datetime.utcfromtimestamp(newmtime))
                    changed.add(docname)
                    continue
                for dep in self.dependencies[docname]:
                    try:
                        deppath = path.join(self.srcdir, dep)
                        if not path.isfile(deppath):
                            changed.add(docname)
                            break
                        depmtime = path.getmtime(deppath)
                        if depmtime > mtime:
                            changed.add(docname)
                            break
                    except OSError:
                        changed.add(docname)
                        break
        return (added, changed, removed)

    def check_dependents(self, app: 'Sphinx', already: Set[str]) -> Generator[str, None, None]:
        to_rewrite: List[str] = []
        for docnames in self.events.emit('env-get-updated', self):
            to_rewrite.extend(docnames)
        for docname in set(to_rewrite):
            if docname not in already:
                yield docname

    def prepare_settings(self, docname: str) -> None:
        """Prepare to set up environment for reading."""
        self.temp_data['docname'] = docname
        self.temp_data['default_role'] = self.config.default_role
        self.temp_data['default_domain'] = self.domains.get(self.config.primary_domain)

    @property
    def docname(self) -> str:
        """Returns the docname of the document currently being parsed."""
        return self.temp_data['docname']

    def new_serialno(self, category: str='') -> int:
        """Return a serial number, e.g. for index entry targets.

        The number is guaranteed to be unique in the current document.
        """
        key = category + 'serialno'
        cur = self.temp_data.get(key, 0)
        self.temp_data[key] = cur + 1
        return cur

    def note_dependency(self, filename: str) -> None:
        """Add *filename* as a dependency of the current document.

        This means that the document will be rebuilt if this file changes.

        *filename* should be absolute or relative to the source directory.
        """
        self.dependencies[self.docname].add(filename)

    def note_included(self, filename: str) -> None:
        """Add *filename* as a included from other document.

        This means the document is not orphaned.

        *filename* should be absolute or relative to the source directory.
        """
        doc = self.path2doc(filename)
        if doc:
            self.included[self.docname].add(doc)

    def note_reread(self) -> None:
        """Add the current document to the list of documents that will
        automatically be re-read at the next build.
        """
        self.reread_always.add(self.docname)

    def get_domain(self, domainname: str) -> Domain:
        """Return the domain instance with the specified name.

        Raises an ExtensionError if the domain is not registered.
        """
        try:
            return self.domains[domainname]
        except KeyError as exc:
            raise ExtensionError(__('Domain %r is not registered') % domainname) from exc

    def get_doctree(self, docname: str) -> nodes.document:
        """Read the doctree for a file from the pickle and return it."""
        filename = path.join(self.doctreedir, docname + '.doctree')
        with open(filename, 'rb') as f:
            doctree = pickle.load(f)
        doctree.settings.env = self
        doctree.reporter = LoggingReporter(self.doc2path(docname))
        return doctree

    def get_and_resolve_doctree(self, docname: str, builder: 'Builder', doctree: Optional[nodes.document]=None, prune_toctrees: bool=True, includehidden: bool=False) -> nodes.document:
        """Read the doctree from the pickle, resolve cross-references and
        toctrees and return it.
        """
        if doctree is None:
            doctree = self.get_doctree(docname)
        self.apply_post_transforms(doctree, docname)
        for toctreenode in doctree.findall(addnodes.toctree):
            result = TocTree(self).resolve(docname, builder, toctreenode, prune=prune_toctrees, includehidden=includehidden)
            if result is None:
                toctreenode.replace_self([])
            else:
                toctreenode.replace_self(result)
        return doctree

    def resolve_toctree(self, docname: str, builder: 'Builder', toctree: addnodes.toctree, prune: bool=True, maxdepth: int=0, titles_only: bool=False, collapse: bool=False, includehidden: bool=False) -> Optional[Node]:
        """Resolve a *toctree* node into individual bullet lists with titles
        as items, returning None (if no containing titles are found) or
        a new node.

        If *prune* is True, the tree is pruned to *maxdepth*, or if that is 0,
        to the value of the *maxdepth* option on the *toctree* node.
        If *titles_only* is True, only toplevel document titles will be in the
        resulting tree.
        If *collapse* is True, all branches not containing docname will
        be collapsed.
        """
        return TocTree(self).resolve(docname, builder, toctree, prune, maxdepth, titles_only, collapse, includehidden)

    def resolve_references(self, doctree: nodes.document, fromdocname: str, builder: 'Builder') -> None:
        self.apply_post_transforms(doctree, fromdocname)

    def apply_post_transforms(self, doctree: nodes.document, docname: str) -> None:
        """Apply all post-transforms."""
        try:
            backup = copy(self.temp_data)
            self.temp_data['docname'] = docname
            transformer = SphinxTransformer(doctree)
            transformer.set_environment(self)
            transformer.add_transforms(self.app.registry.get_post_transforms())
            transformer.apply_transforms()
        finally:
            self.temp_data = backup
        self.events.emit('doctree-resolved', doctree, docname)

    def collect_relations(self) -> Dict[str, List[Optional[str]]]:
        traversed = set()

        def traverse_toctree(parent: Optional[str], docname: str) -> Iterator[Tuple[Optional[str], str]]:
            if parent == docname:
                logger.warning(__('self referenced toctree found. Ignored.'), location=docname, type='toc', subtype='circular')
                return
            yield (parent, docname)
            traversed.add(docname)
            for child in self.toctree_includes.get(docname) or []:
                for subparent, subdocname in traverse_toctree(docname, child):
                    if subdocname not in traversed:
                        yield (subparent, subdocname)
                        traversed.add(subdocname)
        relations = {}
        docnames = traverse_toctree(None, self.config.root_doc)
        prevdoc = None
        parent, docname = next(docnames)
        for nextparent, nextdoc in docnames:
            relations[docname] = [parent, prevdoc, nextdoc]
            prevdoc = docname
            docname = nextdoc
            parent = nextparent
        relations[docname] = [parent, prevdoc, None]
        return relations

    def check_consistency(self) -> None:
        """Do consistency checks."""
        included = set().union(*self.included.values())
        for docname in sorted(self.all_docs):
            if docname not in self.files_to_rebuild:
                if docname == self.config.root_doc:
                    continue
                if docname in included:
                    continue
                if 'orphan' in self.metadata[docname]:
                    continue
                logger.warning(__("document isn't included in any toctree"), location=docname)
        for domain in self.domains.values():
            domain.check_consistency()
        self.events.emit('env-check-consistency', self)