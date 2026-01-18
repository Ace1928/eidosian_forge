import markdown
import markdown.inlinepatterns
import xml.etree.ElementTree as etree
class AutomailExtension(markdown.Extension):
    """
    An extension that turns email addresses into links.
    """

    def extendMarkdown(self, md):
        md.inlinePatterns.register(AutomailPattern(MAIL_RE, md), 'gfm-automail', 100)