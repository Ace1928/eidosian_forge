from within Jinja templates.
from html import escape
from warnings import warn
from traitlets import Dict, observe
from nbconvert.utils.base import NbConvertBase
def _pygments_highlight(source, output_formatter, language='ipython', metadata=None):
    """
    Return a syntax-highlighted version of the input source

    Parameters
    ----------
    source : str
        source of the cell to highlight
    output_formatter : Pygments formatter
    language : str
        language to highlight the syntax of
    metadata : NotebookNode cell metadata
        metadata of the cell to highlight
    """
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound
    if language.startswith('ipython') and metadata and ('magics_language' in metadata):
        language = metadata['magics_language']
    lexer = None
    if language == 'ipython2':
        try:
            from IPython.lib.lexers import IPythonLexer
        except ImportError:
            warn('IPython lexer unavailable, falling back on Python', stacklevel=2)
            language = 'python'
        else:
            lexer = IPythonLexer()
    elif language == 'ipython3':
        try:
            from IPython.lib.lexers import IPython3Lexer
        except ImportError:
            warn('IPython3 lexer unavailable, falling back on Python 3', stacklevel=2)
            language = 'python3'
        else:
            lexer = IPython3Lexer()
    if lexer is None:
        try:
            lexer = get_lexer_by_name(language, stripall=True)
        except ClassNotFound:
            warn('No lexer found for language %r. Treating as plain text.' % language, stacklevel=2)
            from pygments.lexers.special import TextLexer
            lexer = TextLexer()
    return highlight(source, lexer, output_formatter)