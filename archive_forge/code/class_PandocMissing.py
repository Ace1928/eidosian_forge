import re
import shutil
import subprocess
import warnings
from io import BytesIO, TextIOWrapper
from nbconvert.utils.version import check_version
from .exceptions import ConversionException
class PandocMissing(ConversionException):
    """Exception raised when Pandoc is missing."""

    def __init__(self, *args, **kwargs):
        """Initialize the exception."""
        super().__init__("Pandoc wasn't found.\nPlease check that pandoc is installed:\nhttps://pandoc.org/installing.html")