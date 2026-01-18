import re
import datetime
import numpy as np
import csv
import ctypes
def _loadarff(ofile):
    try:
        rel, attr = read_header(ofile)
    except ValueError as e:
        msg = 'Error while parsing header, error was: ' + str(e)
        raise ParseArffError(msg) from e
    hasstr = False
    for a in attr:
        if isinstance(a, StringAttribute):
            hasstr = True
    meta = MetaData(rel, attr)
    if hasstr:
        raise NotImplementedError('String attributes not supported yet, sorry')
    ni = len(attr)

    def generator(row_iter, delim=','):
        elems = list(range(ni))
        dialect = None
        for raw in row_iter:
            if r_comment.match(raw) or r_empty.match(raw):
                continue
            row, dialect = split_data_line(raw, dialect)
            yield tuple([attr[i].parse_data(row[i]) for i in elems])
    a = list(generator(ofile))
    data = np.array(a, [(a.name, a.dtype) for a in attr])
    return (data, meta)