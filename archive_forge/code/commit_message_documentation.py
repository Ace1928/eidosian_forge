import os
import re
import subprocess  # nosec
from hacking import core
Check git commit message length.

    HACKING recommends commit titles 50 chars or less, but enforces
    a 72 character limit

    S365 Title limited to 72 chars
    