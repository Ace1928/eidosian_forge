from pathlib import Path
import json
import os
import pytest
from referencing import Registry
from referencing.exceptions import Unresolvable
import referencing.jsonschema
class SuiteNotFound(Exception):

    def __str__(self):
        return 'Cannot find the referencing suite. Set the REFERENCING_SUITE environment variable to the path to the suite, or run the test suite from alongside a full checkout of the git repository.'