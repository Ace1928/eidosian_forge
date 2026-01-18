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
class NodeReplace(NodeMaker):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return x.replace(self.a, self.b)