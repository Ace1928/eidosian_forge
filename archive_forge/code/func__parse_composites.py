from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def _parse_composites(fh):
    """
    Parse the given filehandle for composites information return them as a
    dict.

    It is assumed that the file cursor is on the line behind 'StartComposites'.

    Returns
    -------
    dict
        A dict mapping composite character names to a parts list. The parts
        list is a list of `.CompositePart` entries describing the parts of
        the composite.

    Examples
    --------
    A composite definition line::

      CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;

    will be represented as::

      composites['Aacute'] = [CompositePart(name='A', dx=0, dy=0),
                              CompositePart(name='acute', dx=160, dy=170)]

    """
    composites = {}
    for line in fh:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(b'EndComposites'):
            return composites
        vals = line.split(b';')
        cc = vals[0].split()
        name, _num_parts = (cc[1], _to_int(cc[2]))
        pccParts = []
        for s in vals[1:-1]:
            pcc = s.split()
            part = CompositePart(pcc[1], _to_float(pcc[2]), _to_float(pcc[3]))
            pccParts.append(part)
        composites[name] = pccParts
    raise RuntimeError('Bad composites parse')