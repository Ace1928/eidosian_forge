from collections import namedtuple
import textwrap
def _validate_io_engine(value):
    if value is not None:
        if value not in ('pyogrio', 'fiona'):
            raise ValueError(f"Expected 'pyogrio' or 'fiona', got '{value}'")