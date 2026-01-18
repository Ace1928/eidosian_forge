from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.document_renderers import renderer
class MarkdownRenderer(renderer.Renderer):
    """Renders markdown to markdown."""

    def __init__(self, *args, **kwargs):
        super(MarkdownRenderer, self).__init__(*args, **kwargs)

    def Write(self, text):
        """Writes text to the markdown output.

    Args:
      text: The text to be written to the output.
    """
        self._out.write(text)