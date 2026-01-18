import pathlib
from os import path
from pprint import pformat
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from jinja2 import BaseLoader, FileSystemLoader, TemplateNotFound
from jinja2.environment import Environment
from jinja2.sandbox import SandboxedEnvironment
from jinja2.utils import open_if_exists
from sphinx.application import TemplateBridge
from sphinx.theming import Theme
from sphinx.util import logging
from sphinx.util.osutil import mtimes_of_files
class BuiltinTemplateLoader(TemplateBridge, BaseLoader):
    """
    Interfaces the rendering environment of jinja2 for use in Sphinx.
    """

    def init(self, builder: 'Builder', theme: Optional[Theme]=None, dirs: Optional[List[str]]=None) -> None:
        if theme:
            pathchain = theme.get_theme_dirs()
            loaderchain = pathchain + [path.join(p, '..') for p in pathchain]
        elif dirs:
            pathchain = list(dirs)
            loaderchain = list(dirs)
        else:
            pathchain = []
            loaderchain = []
        self.templatepathlen = len(builder.config.templates_path)
        if builder.config.templates_path:
            cfg_templates_path = [path.join(builder.confdir, tp) for tp in builder.config.templates_path]
            pathchain[0:0] = cfg_templates_path
            loaderchain[0:0] = cfg_templates_path
        self.pathchain = pathchain
        self.loaders = [SphinxFileSystemLoader(x) for x in loaderchain]
        use_i18n = builder.app.translator is not None
        extensions = ['jinja2.ext.i18n'] if use_i18n else []
        self.environment = SandboxedEnvironment(loader=self, extensions=extensions)
        self.environment.filters['tobool'] = _tobool
        self.environment.filters['toint'] = _toint
        self.environment.filters['todim'] = _todim
        self.environment.filters['slice_index'] = _slice_index
        self.environment.globals['debug'] = pass_context(pformat)
        self.environment.globals['warning'] = warning
        self.environment.globals['accesskey'] = pass_context(accesskey)
        self.environment.globals['idgen'] = idgen
        if use_i18n:
            self.environment.install_gettext_translations(builder.app.translator)

    def render(self, template: str, context: Dict) -> str:
        return self.environment.get_template(template).render(context)

    def render_string(self, source: str, context: Dict) -> str:
        return self.environment.from_string(source).render(context)

    def newest_template_mtime(self) -> float:
        return max(mtimes_of_files(self.pathchain, '.html'))

    def get_source(self, environment: Environment, template: str) -> Tuple[str, str, Callable]:
        loaders = self.loaders
        if template.startswith('!'):
            loaders = loaders[self.templatepathlen:]
            template = template[1:]
        for loader in loaders:
            try:
                return loader.get_source(environment, template)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(template)