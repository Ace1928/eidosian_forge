import logging
from io import BytesIO
from typing import BinaryIO, Iterator, List, Optional, cast
class CorruptDataError(Exception):
    pass