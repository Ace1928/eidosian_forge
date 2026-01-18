import codecs
import pickle
import time
import warnings
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import DependencyList
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import CONFIG_CHANGED_REASON, CONFIG_OK, BuildEnvironment
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.errors import SphinxError
from sphinx.events import EventManager
from sphinx.locale import __
from sphinx.util import (UnicodeDecodeErrorHandler, get_filetype, import_object, logging,
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import sphinx_domains
from sphinx.util.i18n import CatalogInfo, CatalogRepository, docname_to_domain
from sphinx.util.osutil import SEP, ensuredir, relative_uri, relpath
from sphinx.util.parallel import ParallelTasks, SerialTasks, make_chunks, parallel_available
from sphinx.util.tags import Tags
from sphinx.util.typing import NoneType
from sphinx import directives  # NOQA isort:skip
from sphinx import roles  # NOQA isort:skip
def _read_parallel(self, docnames: List[str], nproc: int) -> None:
    chunks = make_chunks(docnames, nproc)
    progress = status_iterator(chunks, __('reading sources... '), 'purple', len(chunks), self.app.verbosity)
    for docname in docnames:
        self.events.emit('env-purge-doc', self.env, docname)
        self.env.clear_doc(docname)

    def read_process(docs: List[str]) -> bytes:
        self.env.app = self.app
        for docname in docs:
            self.read_doc(docname)
        return pickle.dumps(self.env, pickle.HIGHEST_PROTOCOL)

    def merge(docs: List[str], otherenv: bytes) -> None:
        env = pickle.loads(otherenv)
        self.env.merge_info_from(docs, env, self.app)
        next(progress)
    tasks = ParallelTasks(nproc)
    for chunk in chunks:
        tasks.add_task(read_process, chunk, merge)
    tasks.join()
    logger.info('')