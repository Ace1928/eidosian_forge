from __future__ import annotations
from . import Extension
from ..inlinepatterns import SubstituteTagInlineProcessor
class Nl2BrExtension(Extension):

    def extendMarkdown(self, md):
        """ Add a `SubstituteTagInlineProcessor` to Markdown. """
        br_tag = SubstituteTagInlineProcessor(BR_RE, 'br')
        md.inlinePatterns.register(br_tag, 'nl', 5)