from . import matrix
from . import homology
from .polynomial import Polynomial
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase, processFileDispatch, processMagmaFile
from . import utilities
from string import Template
import signal
import re
import os
import sys
from urllib.request import Request, urlopen
from urllib.request import quote as urlquote
from urllib.error import HTTPError
def _retrieve_solution_file(self, data_url=None, prefer_rur=False, verbose=False):
    url = self._solution_file_url(data_url=data_url, rur=prefer_rur)
    if verbose:
        print('Trying to retrieve solutions from %s ...' % url)
    try:
        return _retrieve_url(url)
    except PtolemyFileMissingError:
        url = self._solution_file_url(data_url=data_url, rur=not prefer_rur)
        if verbose:
            print('Retrieving solutions instead from %s ...:' % url)
        return _retrieve_url(url)