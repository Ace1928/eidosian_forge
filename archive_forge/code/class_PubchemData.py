from collections import namedtuple
import warnings
import urllib.request
from urllib.error import URLError, HTTPError
import json
from io import StringIO, BytesIO
from ase.io import read
class PubchemData:
    """
    a specialized class for entries from the pubchem database
    """

    def __init__(self, atoms, data):
        self.atoms = atoms
        self.data = data

    def get_atoms(self):
        return self.atoms

    def get_pubchem_data(self):
        return self.data