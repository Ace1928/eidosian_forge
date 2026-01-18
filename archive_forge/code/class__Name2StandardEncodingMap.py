import os, sys
from reportlab.rl_config import T1SearchPath
from reportlab.lib.utils import rl_isfile
from reportlab.pdfbase._fontdata_enc_winansi import WinAnsiEncoding
from reportlab.pdfbase._fontdata_enc_macroman import MacRomanEncoding
from reportlab.pdfbase._fontdata_enc_standard import StandardEncoding
from reportlab.pdfbase._fontdata_enc_symbol import SymbolEncoding
from reportlab.pdfbase._fontdata_enc_zapfdingbats import ZapfDingbatsEncoding
from reportlab.pdfbase._fontdata_enc_pdfdoc import PDFDocEncoding
from reportlab.pdfbase._fontdata_enc_macexpert import MacExpertEncoding
import reportlab.pdfbase._fontdata_widths_courier
import reportlab.pdfbase._fontdata_widths_courierbold
import reportlab.pdfbase._fontdata_widths_courieroblique
import reportlab.pdfbase._fontdata_widths_courierboldoblique
import reportlab.pdfbase._fontdata_widths_helvetica
import reportlab.pdfbase._fontdata_widths_helveticabold
import reportlab.pdfbase._fontdata_widths_helveticaoblique
import reportlab.pdfbase._fontdata_widths_helveticaboldoblique
import reportlab.pdfbase._fontdata_widths_timesroman
import reportlab.pdfbase._fontdata_widths_timesbold
import reportlab.pdfbase._fontdata_widths_timesitalic
import reportlab.pdfbase._fontdata_widths_timesbolditalic
import reportlab.pdfbase._fontdata_widths_symbol
import reportlab.pdfbase._fontdata_widths_zapfdingbats
from reportlab.rl_config import register_reset
class _Name2StandardEncodingMap(dict):
    """Trivial fake dictionary with some [] magic"""
    _XMap = {'winansi': 'WinAnsiEncoding', 'macroman': 'MacRomanEncoding', 'standard': 'StandardEncoding', 'symbol': 'SymbolEncoding', 'zapfdingbats': 'ZapfDingbatsEncoding', 'pdfdoc': 'PDFDocEncoding', 'macexpert': 'MacExpertEncoding'}

    def __setitem__(self, x, v):
        y = x.lower()
        if y[-8:] == 'encoding':
            y = y[:-8]
        y = self._XMap[y]
        if y in self:
            raise IndexError('Encoding %s is already set' % y)
        dict.__setitem__(self, y, v)

    def __getitem__(self, x):
        y = x.lower()
        if y[-8:] == 'encoding':
            y = y[:-8]
        y = self._XMap[y]
        return dict.__getitem__(self, y)