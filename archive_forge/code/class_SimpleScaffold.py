import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pecan.tests import PecanTestCase
class SimpleScaffold(PecanScaffold):
    _scaffold_dir = ('pecan', os.path.join('tests', 'scaffold_fixtures', 'simple'))