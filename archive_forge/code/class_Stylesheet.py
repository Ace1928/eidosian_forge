import html
import os
import posixpath
import re
import sys
import warnings
from datetime import datetime
from os import path
from typing import IO, Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type
from urllib.parse import quote
import docutils.readers.doctree
from docutils import nodes
from docutils.core import Publisher
from docutils.frontend import OptionParser
from docutils.io import DocTreeInput, StringOutput
from docutils.nodes import Node
from docutils.utils import relative_path
from sphinx import __display_version__, package_dir
from sphinx import version_info as sphinx_version
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx70Warning, deprecated_alias
from sphinx.domains import Domain, Index, IndexEntry
from sphinx.environment import BuildEnvironment
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.environment.adapters.toctree import TocTree
from sphinx.errors import ConfigError, ThemeError
from sphinx.highlighting import PygmentsBridge
from sphinx.locale import _, __
from sphinx.search import js_index
from sphinx.theming import HTMLThemeFactory
from sphinx.util import isurl, logging, md5, progress_message, status_iterator
from sphinx.util.docutils import new_document
from sphinx.util.fileutil import copy_asset
from sphinx.util.i18n import format_date
from sphinx.util.inventory import InventoryFile
from sphinx.util.matching import DOTFILES, Matcher, patmatch
from sphinx.util.osutil import copyfile, ensuredir, os_path, relative_uri
from sphinx.util.tags import Tags
from sphinx.writers.html import HTMLTranslator, HTMLWriter
from sphinx.writers.html5 import HTML5Translator
import sphinxcontrib.serializinghtml  # NOQA
import sphinx.builders.dirhtml  # NOQA
import sphinx.builders.singlehtml  # NOQA
class Stylesheet(str):
    """A metadata of stylesheet.

    To keep compatibility with old themes, an instance of stylesheet behaves as
    its filename (str).
    """
    attributes: Dict[str, str] = None
    filename: str = None
    priority: int = None

    def __new__(cls, filename: str, *args: str, priority: int=500, **attributes: Any) -> 'Stylesheet':
        self = str.__new__(cls, filename)
        self.filename = filename
        self.priority = priority
        self.attributes = attributes
        self.attributes.setdefault('rel', 'stylesheet')
        self.attributes.setdefault('type', 'text/css')
        if args:
            self.attributes['rel'] = args[0]
            self.attributes['title'] = args[1]
        return self