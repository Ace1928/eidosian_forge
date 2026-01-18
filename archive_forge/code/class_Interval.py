import argparse
import ast
import re
import sys
class Interval(object):
    """Represents an interval between a start and end value."""

    def __init__(self, start, start_included, end, end_included):
        self.start = start
        self.start_included = start_included
        self.end = end
        self.end_included = end_included

    def contains(self, value):
        if value < self.start or (value == self.start and (not self.start_included)):
            return False
        if value > self.end or (value == self.end and (not self.end_included)):
            return False
        return True

    def __eq__(self, other):
        return self.start == other.start and self.start_included == other.start_included and (self.end == other.end) and (self.end_included == other.end_included)