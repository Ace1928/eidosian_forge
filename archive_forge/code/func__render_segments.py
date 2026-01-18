from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence
from . import get_console
from .segment import Segment
from .terminal_theme import DEFAULT_TERMINAL_THEME
def _render_segments(segments: Iterable[Segment]) -> str:

    def escape(text: str) -> str:
        """Escape html."""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    fragments: List[str] = []
    append_fragment = fragments.append
    theme = DEFAULT_TERMINAL_THEME
    for text, style, control in Segment.simplify(segments):
        if control:
            continue
        text = escape(text)
        if style:
            rule = style.get_html_style(theme)
            text = f'<span style="{rule}">{text}</span>' if rule else text
            if style.link:
                text = f'<a href="{style.link}" target="_blank">{text}</a>'
        append_fragment(text)
    code = ''.join(fragments)
    html = JUPYTER_HTML_FORMAT.format(code=code)
    return html