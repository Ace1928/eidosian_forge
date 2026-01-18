import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_pe(record, value):
    pe = value.split(':')
    record.protein_existence = int(pe[0])