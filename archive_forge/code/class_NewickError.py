import re
from io import StringIO
from Bio.Phylo import Newick
class NewickError(Exception):
    """Exception raised when Newick object construction cannot continue."""