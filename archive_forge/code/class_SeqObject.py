from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class SeqObject(EvalStringObject):

    def getOp(self, tuple, engine):
        from reportlab.lib.sequencer import getSequencer
        globalsequencer = getSequencer()
        attr = self.attdict
        if 'template' in attr:
            templ = attr['template']
            op = self.op = templ % globalsequencer
            return op
        elif 'id' in attr:
            id = attr['id']
        else:
            id = None
        op = self.op = globalsequencer.nextf(id)
        return op