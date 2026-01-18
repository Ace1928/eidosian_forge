from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
def _stretchCommands(self, n, cmds, oldrowcount):
    """Stretches the commands when a row is split

        The row start is sr, the row end is er.

         sr   | er  | result
        ---------------------------------------------------------------------
          <n  |  <n | Do nothing.
              | >=n | A command that spans the break, extend end.
        ---------------------------------------------------------------------
         ==n  | ==n | Zero height. Extend the end, unless it's a LINEABOVE
              |     | commands, it's between rows so do nothing.
              |     | For LINEBELOW increase both.
              |  >n | A command that spans the break, extend end.
        ---------------------------------------------------------------------
          >n  |  >n | This command comes after the break, increase both.
        ---------------------------------------------------------------------

        Summary:
        1. If er > n then increase er
        2. If sr > n then increase sr
        3. If er == n and sr < n, increase er
        4. If er == sr == n and cmd is not line, increase er

        """
    stretched = [].append
    for c in cmds:
        cmd, (sc, sr), (ec, er) = c[0:3]
        if sr in _SPECIALROWS or er in _SPECIALROWS:
            stretched(c)
            continue
        if er < 0:
            er += oldrowcount
        if sr < 0:
            sr += oldrowcount
        if er > n:
            er += 1
        elif er == n:
            if sr < n or (sr == n and cmd != 'LINEABOVE'):
                er += 1
        if sr > n or (sr == n and cmd == 'LINEBELOW'):
            sr += 1
        stretched((cmd, (sc, sr), (ec, er)) + c[3:])
    return stretched.__self__