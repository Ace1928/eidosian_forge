import os
import pickle
import sys
import warnings
from collections import deque
from io import StringIO
from os import path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, TextElement
from docutils.parsers import Parser
from docutils.parsers.rst import Directive, roles
from docutils.transforms import Transform
from pygments.lexer import Lexer
import sphinx
from sphinx import locale, package_dir
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.domains import Domain, Index
from sphinx.environment import BuildEnvironment
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.errors import ApplicationError, ConfigError, VersionRequirementError
from sphinx.events import EventManager
from sphinx.extension import Extension
from sphinx.highlighting import lexer_classes
from sphinx.locale import __
from sphinx.project import Project
from sphinx.registry import SphinxComponentRegistry
from sphinx.roles import XRefRole
from sphinx.theming import Theme
from sphinx.util import docutils, logging, progress_message
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.i18n import CatalogRepository
from sphinx.util.logging import prefixed_warnings
from sphinx.util.osutil import abspath, ensuredir, relpath
from sphinx.util.tags import Tags
from sphinx.util.typing import RoleFunction, TitleGetter
def _init_i18n(self) -> None:
    """Load translated strings from the configured localedirs if enabled in
        the configuration.
        """
    if self.config.language == 'en':
        self.translator, has_translation = locale.init([], None)
    else:
        logger.info(bold(__('loading translations [%s]... ') % self.config.language), nonl=True)
        repo = CatalogRepository(self.srcdir, self.config.locale_dirs, self.config.language, self.config.source_encoding)
        for catalog in repo.catalogs:
            if catalog.domain == 'sphinx' and catalog.is_outdated():
                catalog.write_mo(self.config.language, self.config.gettext_allow_fuzzy_translations)
        locale_dirs: List[Optional[str]] = list(repo.locale_dirs)
        locale_dirs += [None]
        locale_dirs += [path.join(package_dir, 'locale')]
        self.translator, has_translation = locale.init(locale_dirs, self.config.language)
        if has_translation:
            logger.info(__('done'))
        else:
            logger.info(__('not available for built-in messages'))