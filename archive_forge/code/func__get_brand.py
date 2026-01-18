import abc
import math
import re
import warnings
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import new_class
from typing import (
from uuid import UUID
from weakref import WeakSet
from . import errors
from .datetime_parse import parse_date
from .utils import import_string, update_not_none
from .validators import (
@staticmethod
def _get_brand(card_number: str) -> PaymentCardBrand:
    if card_number[0] == '4':
        brand = PaymentCardBrand.visa
    elif 51 <= int(card_number[:2]) <= 55:
        brand = PaymentCardBrand.mastercard
    elif card_number[:2] in {'34', '37'}:
        brand = PaymentCardBrand.amex
    else:
        brand = PaymentCardBrand.other
    return brand