import configparser
from typing import Dict, List, IO, Mapping, Optional
from .default_styles import DEFAULT_STYLES
from .style import Style, StyleType
class ThemeStack:
    """A stack of themes.

    Args:
        theme (Theme): A theme instance
    """

    def __init__(self, theme: Theme) -> None:
        self._entries: List[Dict[str, Style]] = [theme.styles]
        self.get = self._entries[-1].get

    def push_theme(self, theme: Theme, inherit: bool=True) -> None:
        """Push a theme on the top of the stack.

        Args:
            theme (Theme): A Theme instance.
            inherit (boolean, optional): Inherit styles from current top of stack.
        """
        styles: Dict[str, Style]
        styles = {**self._entries[-1], **theme.styles} if inherit else theme.styles.copy()
        self._entries.append(styles)
        self.get = self._entries[-1].get

    def pop_theme(self) -> None:
        """Pop (and discard) the top-most theme."""
        if len(self._entries) == 1:
            raise ThemeStackError('Unable to pop base theme')
        self._entries.pop()
        self.get = self._entries[-1].get