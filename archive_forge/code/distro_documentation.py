import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from typing import (
A version of @property which caches the value.  On access, it calls the
        underlying function and sets the value in `__dict__` so future accesses
        will not re-call the property.
        