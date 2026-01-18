import re
from ._base import DirectiveParser, BaseDirective
class RSTParser(DirectiveParser):
    name = 'rst_directive'

    @staticmethod
    def parse_type(m: re.Match):
        return m.group('type')

    @staticmethod
    def parse_title(m: re.Match):
        return m.group('title')

    @staticmethod
    def parse_content(m: re.Match):
        full_content = m.group(0)
        text = m.group('text')
        pretext = full_content[:-len(text)]
        leading = len(m.group(1)) + 2
        return '\n'.join((line[leading:] for line in text.splitlines())) + '\n'