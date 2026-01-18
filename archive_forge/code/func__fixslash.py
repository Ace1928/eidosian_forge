from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
def _fixslash(s: str) -> str:
    """Fix windowslike filename to unixlike - (#ifdef WINDOWS)"""
    s = s.replace('\\', '/')
    if s[0] != '/' and s[1] == ':':
        s = s[2:]
    return s