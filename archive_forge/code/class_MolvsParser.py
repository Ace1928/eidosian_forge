import argparse
import logging
import sys
from rdkit import Chem
from rdkit.Chem.MolStandardize import Standardizer, Validator
class MolvsParser(argparse.ArgumentParser):

    def error(self, message):
        sys.stderr.write('Error: %s\n\n'.encode() % message)
        self.print_help()
        sys.exit(2)