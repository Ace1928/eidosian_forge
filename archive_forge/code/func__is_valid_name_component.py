import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _is_valid_name_component(component):
    allowed = string.ascii_letters + string.digits + '-_.: '
    return component and all((x in allowed for x in component))