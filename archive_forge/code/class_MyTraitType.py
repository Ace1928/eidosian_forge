import os
import sys
import tempfile
import textwrap
import shutil
import subprocess
import unittest
from traits.api import (
from traits.testing.optional_dependencies import requires_numpy
class MyTraitType(TraitType):
    default_value_type = DefaultValue.disallow