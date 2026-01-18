from reportlab.lib import PyFontify
from reportlab.platypus.paragraph import Paragraph, _handleBulletWidth, \
from reportlab.lib.utils import isSeq
from reportlab.platypus.flowables import _dedenter
def _get_split_blParaFunc(self):
    return _split_blPara