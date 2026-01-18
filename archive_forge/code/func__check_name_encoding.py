import re
from io import BytesIO
from .. import errors
def _check_name_encoding(name):
    """Check that 'name' is valid UTF-8.

    This is separate from _check_name because UTF-8 decoding is relatively
    expensive, and we usually want to avoid it.

    :raises InvalidRecordError: if name is not valid UTF-8.
    """
    try:
        name.decode('utf-8')
    except UnicodeDecodeError as e:
        raise InvalidRecordError(str(e))