import types
from typing import Iterator
from typing import List
from typing import Union
Iterate over the names of all modules that can be found in the given
    package, recursively.

        >>> import _pytest
        >>> list(_iter_all_modules(_pytest))
        ['_pytest._argcomplete', '_pytest._code.code', ...]
    