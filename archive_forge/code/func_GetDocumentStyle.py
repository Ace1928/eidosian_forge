from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from prompt_toolkit import styles
from prompt_toolkit import token
def GetDocumentStyle():
    """Return the color styles for the layout."""
    prompt_styles = styles.default_style_extensions
    prompt_styles.update({token.Token.Menu.Completions.Completion.Current: Color(BLUE, GRAY), token.Token.Menu.Completions.Completion: Color(BLUE, DARK_GRAY), token.Token.Toolbar: BOLD, token.Token.Toolbar.Account: BOLD, token.Token.Toolbar.Separator: BOLD, token.Token.Toolbar.Project: BOLD, token.Token.Toolbar.Help: BOLD, token.Token.Prompt: BOLD, token.Token.HSep: Color(GREEN), token.Token.Markdown.Section: BOLD, token.Token.Markdown.Definition: BOLD, token.Token.Markdown.Value: ITALIC, token.Token.Markdown.Truncated: REVERSE, token.Token.Purple: BOLD})
    return styles.PygmentsStyle.from_defaults(style_dict=prompt_styles)