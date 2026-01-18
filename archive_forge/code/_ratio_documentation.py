import sys
from fractions import Fraction
from math import ceil
from typing import cast, List, Optional, Sequence
Distribute an integer total in to parts based on ratios.

    Args:
        total (int): The total to divide.
        ratios (List[int]): A list of integer ratios.
        minimums (List[int]): List of minimum values for each slot.

    Returns:
        List[int]: A list of integers guaranteed to sum to total.
    