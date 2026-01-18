import configparser
from os import path
from typing import Dict, Optional
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging
class ThemeFactory:
    """A factory class for LaTeX Themes."""

    def __init__(self, app: Sphinx) -> None:
        self.themes: Dict[str, Theme] = {}
        self.theme_paths = [path.join(app.srcdir, p) for p in app.config.latex_theme_path]
        self.config = app.config
        self.load_builtin_themes(app.config)

    def load_builtin_themes(self, config: Config) -> None:
        """Load built-in themes."""
        self.themes['manual'] = BuiltInTheme('manual', config)
        self.themes['howto'] = BuiltInTheme('howto', config)

    def get(self, name: str) -> Theme:
        """Get a theme for given *name*."""
        if name in self.themes:
            theme = self.themes[name]
        else:
            theme = self.find_user_theme(name) or Theme(name)
        theme.update(self.config)
        return theme

    def find_user_theme(self, name: str) -> Optional[Theme]:
        """Find a theme named as *name* from latex_theme_path."""
        for theme_path in self.theme_paths:
            config_path = path.join(theme_path, name, 'theme.conf')
            if path.isfile(config_path):
                try:
                    return UserTheme(name, config_path)
                except ThemeError as exc:
                    logger.warning(exc)
        return None