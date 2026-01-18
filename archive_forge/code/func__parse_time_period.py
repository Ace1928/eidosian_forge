import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
def _parse_time_period(self, time_period, separator='-'):
    """Parse a string with a time period into a tuple of start and end times."""
    start_time, end_time = time_period.split(separator)
    start_time = time(int(start_time[:2]), int(start_time[-2:]))
    end_time = time(int(end_time[:2]), int(end_time[-2:]))
    return (start_time, end_time)