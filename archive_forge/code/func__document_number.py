import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _document_number(self, section, value, path):
    section.write('%s,' % str(value))