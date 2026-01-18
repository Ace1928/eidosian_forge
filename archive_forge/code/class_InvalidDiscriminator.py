from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class InvalidDiscriminator(PydanticValueError):
    code = 'discriminated_union.invalid_discriminator'
    msg_template = 'No match for discriminator {discriminator_key!r} and value {discriminator_value!r} (allowed values: {allowed_values})'

    def __init__(self, *, discriminator_key: str, discriminator_value: Any, allowed_values: Sequence[Any]) -> None:
        super().__init__(discriminator_key=discriminator_key, discriminator_value=discriminator_value, allowed_values=', '.join(map(repr, allowed_values)))