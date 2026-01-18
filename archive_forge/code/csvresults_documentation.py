from __future__ import annotations
import codecs
import csv
from typing import IO, Dict, List, Optional, Union
from rdflib.plugins.sparql.processor import SPARQLResult
from rdflib.query import Result, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable


This module implements a parser and serializer for the CSV SPARQL result
formats

http://www.w3.org/TR/sparql11-results-csv-tsv/

