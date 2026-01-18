import os
import re
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import (
from ._doc_common import PdfDocCommon, convert_to_int
from ._encryption import Encryption, PasswordType
from ._page import PageObject
from ._utils import (
from .constants import TrailerKeys as TK
from .errors import (
from .generic import (
from .xmp import XmpInformation
@property
def _ID(self) -> Optional[ArrayObject]:
    """
        Provide access to "/ID". standardized with PdfWriter.

        Returns:
            /ID array ; None if the entry does not exists
        """
    id = self.trailer.get(TK.ID, None)
    return None if id is None else cast(ArrayObject, id.get_object())