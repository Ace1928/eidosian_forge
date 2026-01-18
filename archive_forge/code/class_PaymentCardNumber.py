from __future__ import annotations as _annotations
import base64
import dataclasses as _dataclasses
import re
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import (
from uuid import UUID
import annotated_types
from annotated_types import BaseMetadata, MaxLen, MinLen
from pydantic_core import CoreSchema, PydanticCustomError, core_schema
from typing_extensions import Annotated, Literal, Protocol, TypeAlias, TypeAliasType, deprecated
from ._internal import (
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .errors import PydanticUserError
from .json_schema import JsonSchemaValue
from .warnings import PydanticDeprecatedSince20
from pydantic import BaseModel, PositiveInt, ValidationError
from pydantic import BaseModel, NegativeInt, ValidationError
from pydantic import BaseModel, NonPositiveInt, ValidationError
from pydantic import BaseModel, NonNegativeInt, ValidationError
from pydantic import BaseModel, StrictInt, ValidationError
from pydantic import BaseModel, PositiveFloat, ValidationError
from pydantic import BaseModel, NegativeFloat, ValidationError
from pydantic import BaseModel, NonPositiveFloat, ValidationError
from pydantic import BaseModel, NonNegativeFloat, ValidationError
from pydantic import BaseModel, StrictFloat, ValidationError
from pydantic import BaseModel, FiniteFloat
import uuid
from pydantic import UUID1, BaseModel
import uuid
from pydantic import UUID3, BaseModel
import uuid
from pydantic import UUID4, BaseModel
import uuid
from pydantic import UUID5, BaseModel
from pathlib import Path
from pydantic import BaseModel, FilePath, ValidationError
from pathlib import Path
from pydantic import BaseModel, DirectoryPath, ValidationError
from pydantic import Base64Bytes, BaseModel, ValidationError
from pydantic import Base64Str, BaseModel, ValidationError
from pydantic import Base64UrlBytes, BaseModel
from pydantic import Base64UrlStr, BaseModel
@deprecated('The `PaymentCardNumber` class is deprecated, use `pydantic_extra_types` instead. See https://docs.pydantic.dev/latest/api/pydantic_extra_types_payment/#pydantic_extra_types.payment.PaymentCardNumber.', category=PydanticDeprecatedSince20)
class PaymentCardNumber(str):
    """Based on: https://en.wikipedia.org/wiki/Payment_card_number."""
    strip_whitespace: ClassVar[bool] = True
    min_length: ClassVar[int] = 12
    max_length: ClassVar[int] = 19
    bin: str
    last4: str
    brand: PaymentCardBrand

    def __init__(self, card_number: str):
        self.validate_digits(card_number)
        card_number = self.validate_luhn_check_digit(card_number)
        self.bin = card_number[:6]
        self.last4 = card_number[-4:]
        self.brand = self.validate_brand(card_number)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(cls.validate, core_schema.str_schema(min_length=cls.min_length, max_length=cls.max_length, strip_whitespace=cls.strip_whitespace))

    @classmethod
    def validate(cls, __input_value: str, _: core_schema.ValidationInfo) -> PaymentCardNumber:
        """Validate the card number and return a `PaymentCardNumber` instance."""
        return cls(__input_value)

    @property
    def masked(self) -> str:
        """Mask all but the last 4 digits of the card number.

        Returns:
            A masked card number string.
        """
        num_masked = len(self) - 10
        return f'{self.bin}{'*' * num_masked}{self.last4}'

    @classmethod
    def validate_digits(cls, card_number: str) -> None:
        """Validate that the card number is all digits."""
        if not card_number.isdigit():
            raise PydanticCustomError('payment_card_number_digits', 'Card number is not all digits')

    @classmethod
    def validate_luhn_check_digit(cls, card_number: str) -> str:
        """Based on: https://en.wikipedia.org/wiki/Luhn_algorithm."""
        sum_ = int(card_number[-1])
        length = len(card_number)
        parity = length % 2
        for i in range(length - 1):
            digit = int(card_number[i])
            if i % 2 == parity:
                digit *= 2
            if digit > 9:
                digit -= 9
            sum_ += digit
        valid = sum_ % 10 == 0
        if not valid:
            raise PydanticCustomError('payment_card_number_luhn', 'Card number is not luhn valid')
        return card_number

    @staticmethod
    def validate_brand(card_number: str) -> PaymentCardBrand:
        """Validate length based on BIN for major brands:
        https://en.wikipedia.org/wiki/Payment_card_number#Issuer_identification_number_(IIN).
        """
        if card_number[0] == '4':
            brand = PaymentCardBrand.visa
        elif 51 <= int(card_number[:2]) <= 55:
            brand = PaymentCardBrand.mastercard
        elif card_number[:2] in {'34', '37'}:
            brand = PaymentCardBrand.amex
        else:
            brand = PaymentCardBrand.other
        required_length: None | int | str = None
        if brand in PaymentCardBrand.mastercard:
            required_length = 16
            valid = len(card_number) == required_length
        elif brand == PaymentCardBrand.visa:
            required_length = '13, 16 or 19'
            valid = len(card_number) in {13, 16, 19}
        elif brand == PaymentCardBrand.amex:
            required_length = 15
            valid = len(card_number) == required_length
        else:
            valid = True
        if not valid:
            raise PydanticCustomError('payment_card_number_brand', 'Length for a {brand} card must be {required_length}', {'brand': brand, 'required_length': required_length})
        return brand