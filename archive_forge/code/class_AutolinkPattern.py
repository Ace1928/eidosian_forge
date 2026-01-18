import re
import markdown
import xml.etree.ElementTree as etree
class AutolinkPattern(markdown.inlinepatterns.Pattern):

    def handleMatch(self, m):
        el = etree.Element('a')
        href = m.group(2)
        if not PROTOCOL_RE.match(href):
            href = 'http://%s' % href
        el.set('href', self.unescape(href))
        el.text = markdown.util.AtomicString(m.group(2))
        return el