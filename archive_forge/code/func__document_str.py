import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _document_str(self, section, value, path):
    safe_value = escape_controls(value)
    section.write(f"'{safe_value}',")