from __future__ import annotations
import codecs
import configparser
import csv
import datetime
import fileinput
import getopt
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
import rdflib
from rdflib.namespace import RDF, RDFS, split_uri
from rdflib.term import URIRef
from the headers
class NodeBool(NodeLiteral):

    def __call__(self, x):
        if not self.f:
            return rdflib.Literal(bool(x))
        if callable(self.f):
            return rdflib.Literal(bool(self.f(x)))
        raise Exception('Function passed to bool is not callable')

    def range(self):
        return rdflib.XSD.bool