import re
from io import BytesIO
from .. import errors
def _check_name(name):
    """Do some basic checking of 'name'.

    At the moment, this just checks that there are no whitespace characters in a
    name.

    :raises InvalidRecordError: if name is not valid.
    :seealso: _check_name_encoding
    """
    if _whitespace_re.search(name) is not None:
        raise InvalidRecordError('{!r} is not a valid name.'.format(name))