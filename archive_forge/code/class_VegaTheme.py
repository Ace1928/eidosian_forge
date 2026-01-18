from typing import Dict, Union, Final
from ...utils.theme import ThemeRegistry
class VegaTheme:
    """Implementation of a builtin vega theme."""

    def __init__(self, theme: str) -> None:
        self.theme = theme

    def __call__(self) -> Dict[str, Dict[str, Dict[str, Union[str, int]]]]:
        return {'usermeta': {'embedOptions': {'theme': self.theme}}, 'config': {'view': {'continuousWidth': 300, 'continuousHeight': 300}}}

    def __repr__(self) -> str:
        return 'VegaTheme({!r})'.format(self.theme)