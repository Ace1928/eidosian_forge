import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _start_nested_value(self, section, start):
    section.write(start)
    section.style.indent()
    section.style.indent()