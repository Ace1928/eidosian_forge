import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
class _HasVersionFieldProtocol(Protocol):

    def __getitem__(self, s):
        pass

    def __setitem__(self, s, v):
        pass