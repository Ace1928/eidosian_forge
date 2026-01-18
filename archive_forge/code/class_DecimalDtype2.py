import datetime
import decimal
import re
import numpy as np
import pytest
import pytz
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import register_extension_dtype
from pandas.arrays import (
from pandas.core.arrays import (
from pandas.tests.extension.decimal import (
@register_extension_dtype
class DecimalDtype2(DecimalDtype):
    name = 'decimal2'

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return DecimalArray2