import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_rx(reference, value):
    value = value.replace(' [NCBI, ExPASy, Israel, Japan]', '')
    warn = False
    if '=' in value:
        cols = value.split('; ')
        cols = [x.strip() for x in cols]
        cols = [x for x in cols if x]
        for col in cols:
            x = col.split('=')
            if len(x) != 2 or x == ('DOI', 'DOI'):
                warn = True
                break
            assert len(x) == 2, f"I don't understand RX line {value}"
            reference.references.append((x[0], x[1].rstrip(';')))
    else:
        cols = value.split('; ')
        if len(cols) != 2:
            warn = True
        else:
            reference.references.append((cols[0].rstrip(';'), cols[1].rstrip('.')))
    if warn:
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn(f'Possibly corrupt RX line {value!r}', BiopythonParserWarning)