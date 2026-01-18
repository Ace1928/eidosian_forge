import hashlib
import os
import random
from operator import attrgetter
import gyp.common
Writes the solution file to disk.

    Raises:
      IndexError: An entry appears multiple times.
    