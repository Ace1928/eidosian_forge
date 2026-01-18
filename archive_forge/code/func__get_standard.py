from decimal import Decimal
from functools import total_ordering
def _get_standard(self):
    return getattr(self, self.STANDARD_UNIT)