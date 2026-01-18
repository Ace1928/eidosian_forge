import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
class O2:
    member = 0

    def toDict(self):
        return {'member': self.member}