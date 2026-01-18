from pygments.style import Style
from pygments.token import (
class JupyterStyle(Style):
    """
    A pygments style using JupyterLab CSS variables.

    The goal is to mimick JupyterLab's codemirror theme.

    Known impossibilities:

    - With pygments, the dot in `foo.bar` is considered an Operator (class: 'o'),
      while in codemirror, it is bare text.
    - With pygments, in both `from foo import bar`, and `foo.bar`, "bar" is
      considered a Name (class: 'n'), while in coremirror, the latter is a property.

Available CSS variables are

  --jp-mirror-editor-keyword-color
  --jp-mirror-editor-atom-color
  --jp-mirror-editor-number-color
  --jp-mirror-editor-def-color
  --jp-mirror-editor-variable-color
  --jp-mirror-editor-variable-2-color
  --jp-mirror-editor-variable-3-color
  --jp-mirror-editor-punctuation-color
  --jp-mirror-editor-property-color
  --jp-mirror-editor-operator-color
  --jp-mirror-editor-comment-color
  --jp-mirror-editor-string-color
  --jp-mirror-editor-string-2-color
  --jp-mirror-editor-meta-color
  --jp-mirror-editor-qualifier-color
  --jp-mirror-editor-builtin-color
  --jp-mirror-editor-bracket-color
  --jp-mirror-editor-tag-color
  --jp-mirror-editor-attribute-color
  --jp-mirror-editor-header-color
  --jp-mirror-editor-quote-color
  --jp-mirror-editor-link-color
  --jp-mirror-editor-error-color
    """
    default_style = ''
    background_color = 'var(--jp-cell-editor-background)'
    highlight_color = 'var(--jp-cell-editor-active-background)'
    styles = {Text: 'var(--jp-mirror-editor-variable-color)', Whitespace: '', Error: 'var(--jp-mirror-editor-error-color)', Other: '', Comment: 'italic var(--jp-mirror-editor-comment-color)', Keyword: 'bold var(--jp-mirror-editor-keyword-color)', Operator: 'bold var(--jp-mirror-editor-operator-color)', Operator.Word: '', Literal: '', Literal.Date: '', String: 'var(--jp-mirror-editor-string-color)', Number: 'var(--jp-mirror-editor-number-color)', Name: '', Generic: '', Punctuation: 'var(--jp-mirror-editor-punctuation-color)'}