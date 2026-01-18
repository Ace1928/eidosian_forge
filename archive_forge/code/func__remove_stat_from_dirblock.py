import errno
from .. import osutils, tests
from . import features
def _remove_stat_from_dirblock(self, dirblock):
    return [info[:3] + info[4:] for info in dirblock]