import os
import shutil
import sys
from ._process_common import getoutputerror, get_output_error_code, process_handler
class FindCmdError(Exception):
    pass