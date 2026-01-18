import numpy as np
from Bio.Align import Alignment, Alignments
from Bio.Align import bigbed, psl
from Bio.Align.bigbed import AutoSQLTable, Field
from Bio.Seq import Seq, reverse_complement, UndefinedSequenceError
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, Location
from Bio.SeqIO.InsdcIO import _insdc_location_string
Write the file.