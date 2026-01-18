from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence
from . import get_console
from .segment import Segment
from .terminal_theme import DEFAULT_TERMINAL_THEME
class JupyterMixin:
    """Add to an Rich renderable to make it render in Jupyter notebook."""
    __slots__ = ()

    def _repr_mimebundle_(self: 'ConsoleRenderable', include: Sequence[str], exclude: Sequence[str], **kwargs: Any) -> Dict[str, str]:
        console = get_console()
        segments = list(console.render(self, console.options))
        html = _render_segments(segments)
        text = console._render_buffer(segments)
        data = {'text/plain': text, 'text/html': html}
        if include:
            data = {k: v for k, v in data.items() if k in include}
        if exclude:
            data = {k: v for k, v in data.items() if k not in exclude}
        return data