import os, sys, datetime, re
from rdflib import Graph
from ..utils import create_file_name
from . import VocabCachingInfo
import pickle
def _give_preference_path(self):
    """
        Find the vocab cache directory.
        """
    from ...pyRdfa import CACHE_DIR_VAR
    if CACHE_DIR_VAR in os.environ:
        return os.environ[CACHE_DIR_VAR]
    else:
        platform = sys.platform
        if platform in self.architectures:
            system = self.architectures[platform]
        else:
            system = 'unix'
        if system == 'win':
            app_data = os.path.expandvars('%APPDATA%')
            return os.path.join(app_data, self.preference_path[system])
        else:
            return os.path.join(os.path.expanduser('~'), self.preference_path[system])