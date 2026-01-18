import json
import os
import re
import subprocess
import sys
from typing import List, Optional, Set
Tries to pull a package name from the line.

    Used to keep track of what the currently-installing package is,
    in case an error message isn't on the same line as the package
    