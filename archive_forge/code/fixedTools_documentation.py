from .roundTools import otRound, nearestMultipleShortestRepr
import logging
Ensure a table version number is fixed-point.

    Args:
            value (str): a candidate table version number.

    Returns:
            int: A table version number, possibly corrected to fixed-point.
    