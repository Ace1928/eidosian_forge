from collections import defaultdict
from functools import total_ordering
import enum

        Args
        ----
        - callback: callable or None
            It is called for each new casting rule with
            (from_type, to_type, castrel).
        