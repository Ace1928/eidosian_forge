import sys
from osc_lib.command import command
class YamlFormat(RawFormat):

    @property
    def formatter_default(self):
        return 'yaml'