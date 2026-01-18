import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def _psNextPage(self):
    """advance to next page of document.  """
    self.psEndPage()
    self.pageNum = self.pageNum + 1
    self.psBeginPage()