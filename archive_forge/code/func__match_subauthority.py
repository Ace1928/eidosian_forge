from collections import namedtuple
from . import compat
from . import exceptions
from . import misc
from . import normalizers
from . import uri
def _match_subauthority(self):
    return misc.ISUBAUTHORITY_MATCHER.match(self.authority)