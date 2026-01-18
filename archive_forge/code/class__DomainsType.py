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
class _DomainsType(MutableMapping[str, Domain]):

    @overload
    def __getitem__(self, key: Literal['c']) -> CDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['cpp']) -> CPPDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['changeset']) -> ChangeSetDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['citation']) -> CitationDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['index']) -> IndexDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['js']) -> JavaScriptDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['math']) -> MathDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['py']) -> PythonDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['rst']) -> ReSTDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['std']) -> StandardDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['duration']) -> DurationDomain:
        ...

    @overload
    def __getitem__(self, key: Literal['todo']) -> TodoDomain:
        ...

    @overload
    def __getitem__(self, key: str) -> Domain:
        ...

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError