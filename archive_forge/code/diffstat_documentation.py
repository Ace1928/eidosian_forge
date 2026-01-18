import re
import sys
from typing import List, Optional, Tuple
Generate summary statistics from a git style diff ala
       (git diff tag1 tag2 --stat).

    Args:
      lines: list of byte string "lines" from the diff to be parsed
      max_width: maximum line length for generating the summary
                 statistics (default 80)
    Returns: A byte string that lists the changed files with change
             counts and histogram.
    