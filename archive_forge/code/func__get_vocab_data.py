import os, sys, datetime, re
from rdflib import Graph
from ..utils import create_file_name
from . import VocabCachingInfo
import pickle
def _get_vocab_data(self, verify, newCache=True):
    """Just a macro like function to get the data to be cached"""
    from .process import return_graph
    self.graph, self.expiration_date = return_graph(self.uri, self.options, newCache, verify)
    return self.graph != None