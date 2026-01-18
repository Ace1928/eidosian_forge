from decimal import Decimal
from functools import total_ordering
def _set_standard(self, value):
    setattr(self, self.STANDARD_UNIT, value)