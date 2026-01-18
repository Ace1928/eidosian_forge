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
class NodeSplit(NodeMaker):

    def __init__(self, sep, f):
        self.sep = sep
        self.f = f

    def __call__(self, x):
        if not self.f:
            self.f = rdflib.Literal
        if not callable(self.f):
            raise Exception('Function passed to split is not callable!')
        return [self.f(y.strip()) for y in x.split(self.sep) if y.strip() != '']

    def range(self):
        if self.f and isinstance(self.f, NodeMaker):
            return self.f.range()
        return NodeMaker.range(self)