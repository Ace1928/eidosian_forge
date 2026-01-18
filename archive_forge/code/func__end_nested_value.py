import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _end_nested_value(self, section, end):
    section.style.dedent()
    section.style.dedent()
    section.style.new_line()
    section.write(end)