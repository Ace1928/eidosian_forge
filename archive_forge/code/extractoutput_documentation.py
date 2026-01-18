import json
import os
import sys
from binascii import a2b_base64
from mimetypes import guess_extension
from textwrap import dedent
from traitlets import Set, Unicode
from .base import Preprocessor

        Apply a transformation on each cell,

        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        cell_index : int
            Index of the cell being processed (see base.py)
        