import Fontmapping  # helps by mapping pid font classes to Pyart font names
import pyart
from rdkit.sping.PDF import pdfmetrics
from rdkit.sping.pid import *
def _findExternalFontName(self, font):
    """Attempts to return proper font name.
        PDF uses a standard 14 fonts referred to
        by name. Default to self.defaultFont('Helvetica').
        The dictionary allows a layer of indirection to
        support a standard set of PIDDLE font names."""
    piddle_font_map = {'Times': 'Times', 'times': 'Times', 'Courier': 'Courier', 'courier': 'Courier', 'helvetica': 'Helvetica', 'Helvetica': 'Helvetica', 'symbol': 'Symbol', 'Symbol': 'Symbol', 'monospaced': 'Courier', 'serif': 'Times', 'sansserif': 'Helvetica', 'ZapfDingbats': 'ZapfDingbats', 'zapfdingbats': 'ZapfDingbats', 'arial': 'Helvetica'}
    try:
        face = piddle_font_map[font.face.lower()]
    except Exception:
        return 'Helvetica'
    name = face + '-'
    if font.bold and face in ['Courier', 'Helvetica', 'Times']:
        name = name + 'Bold'
    if font.italic and face in ['Courier', 'Helvetica']:
        name = name + 'Oblique'
    elif font.italic and face == 'Times':
        name = name + 'Italic'
    if name == 'Times-':
        name = name + 'Roman'
    if name[-1] == '-':
        name = name[0:-1]
    return name