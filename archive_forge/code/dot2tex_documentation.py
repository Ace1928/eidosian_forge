import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
Process all snippets of code with TeX and preview.sty

        Results are stored in the texdimlist and texdims class attributes.
        Returns False if preprocessing fails
        