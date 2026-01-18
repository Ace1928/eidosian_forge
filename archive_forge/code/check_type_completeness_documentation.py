from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import trio
import trio.testing
Pyright gives us an object identifier of xx.yy.zz
    This function tries to decompose that into its constituent parts, such that we
    can resolve it, in order to check whether it has a `__doc__` at runtime and
    verifytypes misses it because we're doing overly fancy stuff.
    