import sys
import os
import re
import pkg_resources
from string import Template
class BaseScaffold(PecanScaffold):
    _scaffold_dir = ('pecan', os.path.join('scaffolds', 'base'))