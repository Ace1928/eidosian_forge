import datetime
import decimal
import json
import re
import uuid
from .. import config
from .. import engines
from .. import fixtures
from .. import mock
from ..assertions import eq_
from ..assertions import is_
from ..assertions import ne_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import and_
from ... import ARRAY
from ... import BigInteger
from ... import bindparam
from ... import Boolean
from ... import case
from ... import cast
from ... import Date
from ... import DateTime
from ... import Float
from ... import Integer
from ... import Interval
from ... import JSON
from ... import literal
from ... import literal_column
from ... import MetaData
from ... import null
from ... import Numeric
from ... import select
from ... import String
from ... import testing
from ... import Text
from ... import Time
from ... import TIMESTAMP
from ... import type_coerce
from ... import TypeDecorator
from ... import Unicode
from ... import UnicodeText
from ... import UUID
from ... import Uuid
from ...orm import declarative_base
from ...orm import Session
from ...sql import sqltypes
from ...sql.sqltypes import LargeBinary
from ...sql.sqltypes import PickleType
def _index_fixtures(include_comparison):
    if include_comparison:
        json_elements = []
    else:
        json_elements = [('json', {'foo': 'bar'}), ('json', ['one', 'two', 'three']), (None, {'foo': 'bar'}), (None, ['one', 'two', 'three'])]
    elements = [('boolean', True), ('boolean', False), ('boolean', None), ('string', 'some string'), ('string', None), ('string', 'réve illé'), ('string', 'réve🐍 illé', testing.requires.json_index_supplementary_unicode_element), ('integer', 15), ('integer', 1), ('integer', 0), ('integer', None), ('float', 28.5), ('float', None), ('float', 1234567.89, testing.requires.literal_float_coercion), ('numeric', 1234567.89), ('numeric', 99998969694839.98), ('numeric', 99939.983485848), ('_decimal', decimal.Decimal('1234567.89')), ('_decimal', decimal.Decimal('99998969694839.983485848'), requirements.cast_precision_numerics_many_significant_digits), ('_decimal', decimal.Decimal('99939.983485848'))] + json_elements

    def decorate(fn):
        fn = testing.combinations(*elements, id_='sa')(fn)
        return fn
    return decorate