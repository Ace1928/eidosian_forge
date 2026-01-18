from ebooklib.plugins.base import BasePlugin
from ebooklib.utils import parse_html_string
class BooktypeLinks(BasePlugin):
    NAME = 'Booktype Links'

    def __init__(self, booktype_book):
        self.booktype_book = booktype_book

    def html_before_write(self, book, chapter):
        from lxml import etree
        try:
            from urlparse import urlparse, urljoin
        except ImportError:
            from urllib.parse import urlparse, urljoin
        try:
            tree = parse_html_string(chapter.content)
        except:
            return
        root = tree.getroottree()
        if len(root.find('body')) != 0:
            body = tree.find('body')
            for _link in body.xpath('//a'):
                if _link.get('href', '').find('InsertNoteID') != -1:
                    _ln = _link.get('href', '')
                    i = _ln.find('#')
                    _link.set('href', _ln[i:])
                    continue
                _u = urlparse(_link.get('href', ''))
                if _u.scheme == '':
                    if _u.path != '':
                        _link.set('href', '%s.xhtml' % _u.path)
                    if _u.fragment != '':
                        _link.set('href', urljoin(_link.get('href'), '#%s' % _u.fragment))
                    if _link.get('name') != None:
                        _link.set('id', _link.get('name'))
                        etree.strip_attributes(_link, 'name')
        chapter.content = etree.tostring(tree, pretty_print=True, encoding='utf-8')