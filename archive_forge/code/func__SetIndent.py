from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
def _SetIndent(self, level, indent=0, second_line_indent=None):
    """Sets the markdown list level and indentations.

    Args:
      level: int, The desired markdown list level.
      indent: int, The new indentation.
      second_line_indent: int, The second line indentation. This is subtracted
        from the prevailing indent to decrease the indentation of the next input
        line for this effect:
            SECOND LINE INDENT ON THE NEXT LINE
               PREVAILING INDENT
               ON SUBSEQUENT LINES
    """
    if self._level < level:
        while self._level < level:
            prev_level = self._level
            self._level += 1
            if self._level >= len(self._indent):
                self._indent.append(self.Indent())
            self._indent[self._level].indent = self._indent[prev_level].indent + indent
            if self._level > 1 and self._indent[prev_level].second_line_indent == self._indent[prev_level].indent:
                self._indent[self._level].indent += 1
            self._indent[self._level].second_line_indent = self._indent[self._level].indent
            if second_line_indent is not None:
                self._indent[self._level].second_line_indent -= second_line_indent
    else:
        self._level = level
        if second_line_indent is not None:
            self._indent[self._level].indent = self._indent[self._level].second_line_indent + second_line_indent