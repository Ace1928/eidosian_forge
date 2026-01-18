import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class SingleAlternativeChild(SingleAlternative):

    def alt(self) -> None:
        """Alternative method."""