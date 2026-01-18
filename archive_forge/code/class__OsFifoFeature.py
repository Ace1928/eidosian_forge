import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _OsFifoFeature(Feature):

    def _probe(self):
        return getattr(os, 'mkfifo', None)

    def feature_name(self):
        return 'filesystem fifos'