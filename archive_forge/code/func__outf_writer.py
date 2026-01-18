import csv
import gzip
import json
from nltk.internals import deprecated
def _outf_writer(outfile, encoding, errors, gzip_compress=False):
    if gzip_compress:
        outf = gzip.open(outfile, 'wt', newline='', encoding=encoding, errors=errors)
    else:
        outf = open(outfile, 'w', newline='', encoding=encoding, errors=errors)
    writer = csv.writer(outf)
    return (writer, outf)