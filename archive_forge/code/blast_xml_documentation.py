import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
Adjust output to mimic native BLAST+ XML as much as possible (PRIVATE).