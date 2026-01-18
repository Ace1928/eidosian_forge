import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def _ignore_atime(stat):
    return stat[:7] + stat[8:]