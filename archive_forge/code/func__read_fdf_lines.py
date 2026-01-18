from pathlib import Path
from re import compile
import numpy as np
from ase import Atoms
from ase.utils import reader
from ase.units import Bohr
@reader
def _read_fdf_lines(file):
    lbz = _labelize
    lines = []
    for L in _get_stripped_lines(file):
        w0 = lbz(L.split(None, 1)[0])
        if w0 == '%include':
            fname = L.split(None, 1)[1].strip()
            parent_fname = getattr(file, 'name', None)
            if isinstance(parent_fname, str):
                fname = Path(parent_fname).parent / fname
            lines += _read_fdf_lines(fname)
        elif '<' in L:
            L, fname = L.split('<', 1)
            w = L.split()
            fname = fname.strip()
            if w0 == '%block':
                if len(w) != 2:
                    raise IOError('Bad %%block-statement "%s < %s"' % (L, fname))
                label = lbz(w[1])
                lines.append('%%block %s' % label)
                lines += _get_stripped_lines(open(fname))
                lines.append('%%endblock %s' % label)
            else:
                label = lbz(w[0])
                fdf = read_fdf(fname)
                if label in fdf:
                    if _is_block(fdf[label]):
                        lines.append('%%block %s' % label)
                        lines += [' '.join(x) for x in fdf[label]]
                        lines.append('%%endblock %s' % label)
                    else:
                        lines.append('%s %s' % (label, ' '.join(fdf[label])))
        else:
            lines.append(L)
    return lines