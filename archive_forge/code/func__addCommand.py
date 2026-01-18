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
def _addCommand(self, cmd):
    if cmd[0] in ('BACKGROUND', 'ROWBACKGROUNDS', 'COLBACKGROUNDS'):
        self._bkgrndcmds.append(cmd)
    elif cmd[0] == 'SPAN':
        self._spanCmds.append(cmd)
    elif cmd[0] == 'NOSPLIT':
        self._nosplitCmds.append(cmd)
    elif _isLineCommand(cmd):
        cmd = list(cmd)
        if len(cmd) < 5:
            raise ValueError(f'bad line command {cmd!a}')
        if len(cmd) < 6:
            cmd.append(1)
        else:
            cap = _convert2int(cmd[5], LINECAPS, 0, 2, 'cap', cmd)
            cmd[5] = cap
        if len(cmd) < 7:
            cmd.append(None)
        if len(cmd) < 8:
            cmd.append(1)
        else:
            join = _convert2int(cmd[7], LINEJOINS, 0, 2, 'join', cmd)
            cmd[7] = join
        if len(cmd) < 9:
            cmd.append(1)
        else:
            lineCount = cmd[8]
            if lineCount is None:
                lineCount = 1
                cmd[8] = lineCount
            assert lineCount >= 1
        if len(cmd) < 10:
            cmd.append(cmd[3])
        else:
            space = cmd[9]
            if space is None:
                space = cmd[3]
                cmd[9] = space
        assert len(cmd) == 10
        self._linecmds.append(tuple(cmd))
    elif cmd[0] == 'ROUNDEDCORNERS':
        self._setCornerRadii(cmd[1])
    else:
        (op, (sc, sr), (ec, er)), values = (cmd[:3], cmd[3:])
        if sr in _SPECIALROWS:
            (self._srflcmds if sr[0] == 's' else self._sircmds).append(cmd)
        else:
            sc, ec, sr, er = self.normCellRange(sc, ec, sr, er)
            ec += 1
            for i in range(sr, er + 1):
                for j in range(sc, ec):
                    _setCellStyle(self._cellStyles, i, j, op, values)