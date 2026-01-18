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
class SphinxStandaloneReader(SphinxBaseReader):
    """
    A basic document reader for Sphinx.
    """

    def setup(self, app: 'Sphinx') -> None:
        self.transforms = self.transforms + app.registry.get_transforms()
        super().setup(app)

    def read(self, source: Input, parser: Parser, settings: Values) -> nodes.document:
        self.source = source
        if not self.parser:
            self.parser = parser
        self.settings = settings
        self.input = self.read_source(settings.env)
        self.parse()
        return self.document

    def read_source(self, env: BuildEnvironment) -> str:
        """Read content from source and do post-process."""
        content = self.source.read()
        arg = [content]
        env.events.emit('source-read', env.docname, arg)
        return arg[0]