from __future__ import annotations
from .prettytable import PrettyTable
class ColorTable(PrettyTable):

    def __init__(self, field_names=None, **kwargs) -> None:
        super().__init__(field_names=field_names, **kwargs)
        self.theme = kwargs.get('theme') or Themes.DEFAULT

    @property
    def theme(self) -> Theme:
        return self._theme

    @theme.setter
    def theme(self, value: Theme) -> None:
        self._theme = value
        self.update_theme()

    def update_theme(self) -> None:
        theme = self._theme
        self._vertical_char = theme.vertical_color + theme.vertical_char + RESET_CODE + theme.default_color
        self._horizontal_char = theme.horizontal_color + theme.horizontal_char + RESET_CODE + theme.default_color
        self._junction_char = theme.junction_color + theme.junction_char + RESET_CODE + theme.default_color

    def get_string(self, **kwargs) -> str:
        return super().get_string(**kwargs) + RESET_CODE