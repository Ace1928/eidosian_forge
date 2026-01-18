from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class MissingDiscriminator(PydanticValueError):
    code = 'discriminated_union.missing_discriminator'
    msg_template = 'Discriminator {discriminator_key!r} is missing in value'