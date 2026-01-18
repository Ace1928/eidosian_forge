import shlex
import itertools
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord
def _write_trackline(self, stream, metadata):
    stream.write('track')
    for key, value in metadata.items():
        if key in ('name', 'description', 'frames'):
            pass
        elif key == 'mafDot':
            if value not in ('on', 'off'):
                raise ValueError("mafDot value must be 'on' or 'off' (received '%s')" % value)
        elif key == 'visibility':
            if value not in ('dense', 'pack', 'full'):
                raise ValueError("visibility value must be 'dense', 'pack', or 'full' (received '%s')" % value)
        elif key == 'speciesOrder':
            value = ' '.join(value)
        else:
            continue
        if ' ' in value:
            value = '"%s"' % value
        stream.write(f' {key}={value}')
    stream.write('\n')