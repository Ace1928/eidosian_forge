from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import Name, Assign
class FixStdDevs(BaseFix):
    PATTERN = "\n    power< any* trailer< '.' 'std_devs' > trailer< '(' ')' > >\n    "

    def transform(self, node, results):
        node.children[-1].remove()