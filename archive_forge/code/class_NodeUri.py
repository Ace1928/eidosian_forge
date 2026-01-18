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
class NodeUri(NodeMaker):

    def __init__(self, prefix, class_):
        self.prefix = prefix
        if class_:
            self.class_ = rdflib.URIRef(class_)
        else:
            self.class_ = None

    def __call__(self, x):
        return prefixuri(x, self.prefix, self.class_)

    def range(self):
        return self.class_ or rdflib.RDF.Resource