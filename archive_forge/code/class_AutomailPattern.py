import markdown
import markdown.inlinepatterns
import xml.etree.ElementTree as etree
class AutomailPattern(markdown.inlinepatterns.Pattern):

    def handleMatch(self, m):
        el = etree.Element('a')
        el.set('href', self.unescape('mailto:' + m.group(2)))
        el.text = markdown.util.AtomicString(m.group(2))
        return el