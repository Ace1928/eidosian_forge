import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class FloatOperation(DecimalException, TypeError):
    """Enable stricter semantics for mixing floats and Decimals.

    If the signal is not trapped (default), mixing floats and Decimals is
    permitted in the Decimal() constructor, context.create_decimal() and
    all comparison operators. Both conversion and comparisons are exact.
    Any occurrence of a mixed operation is silently recorded by setting
    FloatOperation in the context flags.  Explicit conversions with
    Decimal.from_float() or context.create_decimal_from_float() do not
    set the flag.

    Otherwise (the signal is trapped), only equality comparisons and explicit
    conversions are silent. All other mixed operations raise FloatOperation.
    """