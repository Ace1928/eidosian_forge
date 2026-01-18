import hashlib
import os
import random
from operator import attrgetter
import gyp.common
class MSVSSolutionEntry:

    def __cmp__(self, other):
        return cmp((self.name, self.get_guid()), (other.name, other.get_guid()))