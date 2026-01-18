import functools
import os
import re
import sys
import warnings
from io import StringIO
from typing import IO, Any, Dict, Generator, List, Optional, Pattern
from xml.etree import ElementTree
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives, roles
from sphinx import application, locale
from sphinx.pycode import ModuleAnalyzer
from sphinx.testing.path import path
from sphinx.util.osutil import relpath
class SphinxTestApp(application.Sphinx):
    """
    A subclass of :class:`Sphinx` that runs on the test root, with some
    better default values for the initialization parameters.
    """
    _status: StringIO
    _warning: StringIO

    def __init__(self, buildername: str='html', srcdir: Optional[path]=None, builddir: Optional[path]=None, freshenv: bool=False, confoverrides: Optional[Dict]=None, status: Optional[IO]=None, warning: Optional[IO]=None, tags: Optional[List[str]]=None, docutilsconf: Optional[str]=None, parallel: int=0) -> None:
        if docutilsconf is not None:
            (srcdir / 'docutils.conf').write_text(docutilsconf)
        if builddir is None:
            builddir = srcdir / '_build'
        confdir = srcdir
        outdir = builddir.joinpath(buildername)
        outdir.makedirs(exist_ok=True)
        doctreedir = builddir.joinpath('doctrees')
        doctreedir.makedirs(exist_ok=True)
        if confoverrides is None:
            confoverrides = {}
        warningiserror = False
        self._saved_path = sys.path[:]
        self._saved_directives = directives._directives.copy()
        self._saved_roles = roles._roles.copy()
        self._saved_nodeclasses = {v for v in dir(nodes.GenericNodeVisitor) if v.startswith('visit_')}
        try:
            super().__init__(srcdir, confdir, outdir, doctreedir, buildername, confoverrides, status, warning, freshenv, warningiserror, tags, parallel=parallel)
        except Exception:
            self.cleanup()
            raise

    def cleanup(self, doctrees: bool=False) -> None:
        ModuleAnalyzer.cache.clear()
        locale.translators.clear()
        sys.path[:] = self._saved_path
        sys.modules.pop('autodoc_fodder', None)
        directives._directives = self._saved_directives
        roles._roles = self._saved_roles
        for method in dir(nodes.GenericNodeVisitor):
            if method.startswith('visit_') and method not in self._saved_nodeclasses:
                delattr(nodes.GenericNodeVisitor, 'visit_' + method[6:])
                delattr(nodes.GenericNodeVisitor, 'depart_' + method[6:])

    def __repr__(self) -> str:
        return '<%s buildername=%r>' % (self.__class__.__name__, self.builder.name)