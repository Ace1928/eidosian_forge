import markdown.inlinepatterns
class StrikethroughExtension(markdown.Extension):
    """
    An extension that adds support for strike-through text between two ``~~``.
    """

    def extendMarkdown(self, md):
        md.inlinePatterns.register(markdown.inlinepatterns.SimpleTagPattern(STRIKE_RE, 'del'), 'gfm-strikethrough', 100)