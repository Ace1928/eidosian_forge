import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
class MolFormatter:
    """Format molecules as images"""

    def __init__(self, orig_formatter=None):
        """Store original formatters (if any)"""
        self.orig_formatter = orig_formatter

    @staticmethod
    def default_formatter(x):
        """Default formatter function"""
        return pprint_thing(x, escape_chars=('\t', '\r', '\n'))

    @staticmethod
    def is_mol(x):
        """Return True if x is a Chem.Mol"""
        return isinstance(x, Mol)

    @classmethod
    def get_formatters(cls, df, orig_formatters):
        """Return an instance of MolFormatter for each column that contains Chem.Mol objects"""
        df_subset = df.select_dtypes('object')
        return {col: cls(orig_formatters.get(col, None)) for col in df_subset.columns[df_subset.applymap(MolFormatter.is_mol).any()]}

    def __call__(self, x):
        """Return x formatted based on its type"""
        if self.is_mol(x):
            return PrintAsImageString(x)
        if callable(self.orig_formatter):
            return self.orig_formatter(x)
        return self.default_formatter(x)