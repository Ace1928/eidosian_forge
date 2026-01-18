import codecs
import warnings
from typing import TYPE_CHECKING, Any, List, Type
import docutils
from docutils import nodes
from docutils.core import Publisher
from docutils.frontend import Values
from docutils.io import FileInput, Input, NullOutput
from docutils.parsers import Parser
from docutils.parsers.rst import Parser as RSTParser
from docutils.readers import standalone
from docutils.transforms import Transform
from docutils.transforms.references import DanglingReferences
from docutils.writers import UnfilteredWriter
from sphinx import addnodes
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment
from sphinx.transforms import (AutoIndexUpgrader, DoctreeReadEvent, FigureAligner,
from sphinx.transforms.i18n import (Locale, PreserveTranslatableMessages,
from sphinx.transforms.references import SphinxDomains
from sphinx.util import UnicodeDecodeErrorHandler, get_filetype, logging
from sphinx.util.docutils import LoggingReporter
from sphinx.versioning import UIDTransform
class SphinxFileInput(FileInput):
    """A basic FileInput for Sphinx."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs['error_handler'] = 'sphinx'
        super().__init__(*args, **kwargs)