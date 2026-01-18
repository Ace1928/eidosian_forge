import struct
from .constants import ER
def _map_error(exc, *errors):
    for error in errors:
        error_map[error] = exc