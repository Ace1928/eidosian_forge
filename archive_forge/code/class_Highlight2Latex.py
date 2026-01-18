from within Jinja templates.
from html import escape
from warnings import warn
from traitlets import Dict, observe
from nbconvert.utils.base import NbConvertBase
class Highlight2Latex(NbConvertBase):
    """Convert highlighted code to latex."""
    extra_formatter_options = Dict({}, help="\n        Extra set of options to control how code is highlighted.\n\n        Passed through to the pygments' LatexFormatter class.\n        See available list in https://pygments.org/docs/formatters/#LatexFormatter\n        ", config=True)

    def __init__(self, pygments_lexer=None, **kwargs):
        """Initialize the converter."""
        self.pygments_lexer = pygments_lexer or 'ipython3'
        super().__init__(**kwargs)

    @observe('default_language')
    def _default_language_changed(self, change):
        warn('Setting default_language in config is deprecated as of 5.0, please use language_info metadata instead.', stacklevel=2)
        self.pygments_lexer = change['new']

    def __call__(self, source, language=None, metadata=None, strip_verbatim=False):
        """
        Return a syntax-highlighted version of the input source as latex output.

        Parameters
        ----------
        source : str
            source of the cell to highlight
        language : str
            language to highlight the syntax of
        metadata : NotebookNode cell metadata
            metadata of the cell to highlight
        strip_verbatim : bool
            remove the Verbatim environment that pygments provides by default
        """
        from pygments.formatters import LatexFormatter
        if not language:
            language = self.pygments_lexer
        latex = _pygments_highlight(source, LatexFormatter(**self.extra_formatter_options), language, metadata)
        if strip_verbatim:
            latex = latex.replace('\\begin{Verbatim}[commandchars=\\\\\\{\\}]' + '\n', '')
            return latex.replace('\n\\end{Verbatim}\n', '')
        return latex