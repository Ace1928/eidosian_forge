import base64
import copy
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from rdflib.query import Result
from rdflib.term import BNode
class SPARQLConnectorException(Exception):
    pass