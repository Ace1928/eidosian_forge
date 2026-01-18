import subprocess
import doctest
import os
import sys
import shutil
import re
import cgi
import rfc822
from io import StringIO
from paste.util import PySourceColor

        Divide the given string into examples and intervening text,
        and return them as a list of alternating Examples and strings.
        Line numbers for the Examples are 0-based.  The optional
        argument `name` is a name identifying this string, and is only
        used for error messages.
        