from collections import OrderedDict
import datetime as dt
import decimal
from io import StringIO
import json
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.string_ import StringDtype
from pandas.core.series import Series
from pandas.tests.extension.date import (
from pandas.tests.extension.decimal.array import (
from pandas.io.json._table_schema import (
class TestTableSchemaType:

    @pytest.mark.parametrize('date_data', [DateArray([dt.date(2021, 10, 10)]), DateArray(dt.date(2021, 10, 10)), Series(DateArray(dt.date(2021, 10, 10)))])
    def test_as_json_table_type_ext_date_array_dtype(self, date_data):
        assert as_json_table_type(date_data.dtype) == 'any'

    def test_as_json_table_type_ext_date_dtype(self):
        assert as_json_table_type(DateDtype()) == 'any'

    @pytest.mark.parametrize('decimal_data', [DecimalArray([decimal.Decimal(10)]), Series(DecimalArray([decimal.Decimal(10)]))])
    def test_as_json_table_type_ext_decimal_array_dtype(self, decimal_data):
        assert as_json_table_type(decimal_data.dtype) == 'number'

    def test_as_json_table_type_ext_decimal_dtype(self):
        assert as_json_table_type(DecimalDtype()) == 'number'

    @pytest.mark.parametrize('string_data', [array(['pandas'], dtype='string'), Series(array(['pandas'], dtype='string'))])
    def test_as_json_table_type_ext_string_array_dtype(self, string_data):
        assert as_json_table_type(string_data.dtype) == 'any'

    def test_as_json_table_type_ext_string_dtype(self):
        assert as_json_table_type(StringDtype()) == 'any'

    @pytest.mark.parametrize('integer_data', [array([10], dtype='Int64'), Series(array([10], dtype='Int64'))])
    def test_as_json_table_type_ext_integer_array_dtype(self, integer_data):
        assert as_json_table_type(integer_data.dtype) == 'integer'

    def test_as_json_table_type_ext_integer_dtype(self):
        assert as_json_table_type(Int64Dtype()) == 'integer'