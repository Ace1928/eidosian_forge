import codecs
import getopt
import sys
import time
import rdflib
from rdflib.util import guess_format

    A main function for tools that read RDF from files given on commandline
    or from STDIN (if stdin parameter is true)
    