import base64
import re
from io import BytesIO
from .... import errors, registry
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....timestamp import format_highres_date, unpack_highres_date
def _get_bundle_header(version):
    return b''.join([BUNDLE_HEADER, version.encode('ascii'), b'\n'])