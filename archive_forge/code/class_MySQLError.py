import struct
from .constants import ER
class MySQLError(Exception):
    """Exception related to operation with MySQL."""