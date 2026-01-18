import contextlib
import doctest
from io import StringIO
import itertools
import os
from os.path import relpath
from pathlib import Path
import re
import shutil
import sys
import textwrap
import traceback
from docutils.parsers.rst import directives, Directive
from docutils.parsers.rst.directives.images import Image
import jinja2  # Sphinx dependency.
from sphinx.errors import ExtensionError
import matplotlib
from matplotlib.backend_bases import FigureManagerBase
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers, cbook
class PlotDirective(Directive):
    """The ``.. plot::`` directive, as documented in the module's docstring."""
    has_content = True
    required_arguments = 0
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {'alt': directives.unchanged, 'height': directives.length_or_unitless, 'width': directives.length_or_percentage_or_unitless, 'scale': directives.nonnegative_int, 'align': Image.align, 'class': directives.class_option, 'include-source': _option_boolean, 'show-source-link': _option_boolean, 'format': _option_format, 'context': _option_context, 'nofigs': directives.flag, 'caption': directives.unchanged}

    def run(self):
        """Run the plot directive."""
        try:
            return run(self.arguments, self.content, self.options, self.state_machine, self.state, self.lineno)
        except Exception as e:
            raise self.error(str(e))