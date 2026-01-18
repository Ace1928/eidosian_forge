from io import BytesIO
from testtools import content, content_type
from testtools.compat import _b
from subunit import chunked
Parser for multi-part [] surrounded MIME typed chunked details.