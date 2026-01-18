import numbers
from typing import AbstractSet, Any, cast, TYPE_CHECKING, TypeVar
from typing_extensions import Self
import sympy
from typing_extensions import Protocol
from cirq import study
from cirq._doc import doc_private
@doc_private
def _resolved_value_(self) -> Any:
    """Returns a resolved value during parameter resolution.

        Use this to mark a custom type as "resolved", instead of requiring
        further parsing like we do with Sympy symbols.
        """