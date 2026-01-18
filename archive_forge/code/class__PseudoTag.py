import re
from lxml import etree, html
class _PseudoTag:

    def __init__(self, contents):
        self.name = 'html'
        self.attrs = []
        self.contents = contents

    def __iter__(self):
        return self.contents.__iter__()