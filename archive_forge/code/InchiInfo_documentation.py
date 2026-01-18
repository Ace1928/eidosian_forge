import logging
import re
from rdkit import Chem
from rdkit.Chem import inchi
 retrieve mobile H (tautomer) information
        return a 2-item tuple containing
        1) Number of mobile hydrogen groups detected. If 0, next item = '' 
        2) List of groups   
        