import sys
from osc_lib.command import command
class JsonFormat(RawFormat):

    @property
    def formatter_default(self):
        return 'json'