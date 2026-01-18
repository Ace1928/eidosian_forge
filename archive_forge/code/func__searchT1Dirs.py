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
def _searchT1Dirs(n, rl_isfile=rl_isfile, T1SearchPath=T1SearchPath):
    assert T1SearchPath != [], 'No Type-1 font search path'
    for d in T1SearchPath:
        f = os.path.join(d, n)
        if rl_isfile(f):
            return f
    return None