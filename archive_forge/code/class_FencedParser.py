import re
from ._base import DirectiveParser, BaseDirective
class FencedParser(DirectiveParser):
    name = 'fenced_directive'

    @staticmethod
    def parse_type(m: re.Match):
        return m.group('type')

    @staticmethod
    def parse_title(m: re.Match):
        return m.group('title')

    @staticmethod
    def parse_content(m: re.Match):
        return m.group('text')