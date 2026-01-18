import sys
import io
import copy
import array
import itertools
import struct
import zlib
from collections import namedtuple
from io import BytesIO
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
Iterate over alignments overlapping the specified chromosome region..

        This method searches the index to find alignments to the specified
        chromosome that fully or partially overlap the chromosome region
        between start and end.

        Arguments:
         - chromosome - chromosome name. If None (default value), include all
           alignments.
         - start      - starting position on the chromosome. If None (default
           value), use 0 as the starting position.
         - end        - end position on the chromosome. If None (default value),
           use the length of the chromosome as the end position.

        