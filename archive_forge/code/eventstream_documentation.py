from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
Closes the underlying streaming body.