from typing import Any
import sys
from typing import _type_check  # type: ignore
class _FlexibleAliasCls:

    def __getitem__(self, args):
        return _FlexibleAliasClsApplied(args[-1])