import binascii, codecs, zlib
from collections import OrderedDict
from reportlab.pdfbase import pdfutils
from reportlab import rl_config
from reportlab.lib.utils import open_for_read, makeFileName, isSeq, isBytes, isUnicode, _digester, isStr, bytestr, annotateException, TimeStamp
from reportlab.lib.rl_accel import escapePDF, fp_str, asciiBase85Encode, asciiBase85Decode
from reportlab.pdfbase import pdfmetrics
from hashlib import md5
from sys import stderr
import re
class ViewerPreferencesPDFDictionary(CheckedPDFDictionary):
    validate = dict(HideToolbar=checkPDFBoolean, HideMenubar=checkPDFBoolean, HideWindowUI=checkPDFBoolean, FitWindow=checkPDFBoolean, CenterWindow=checkPDFBoolean, DisplayDocTitle=checkPDFBoolean, NonFullScreenPageMode=checkPDFNames(*'UseNone UseOutlines UseThumbs UseOC'.split()), Direction=checkPDFNames(*'L2R R2L'.split()), ViewArea=checkPDFNames(*'MediaBox CropBox BleedBox TrimBox ArtBox'.split()), ViewClip=checkPDFNames(*'MediaBox CropBox BleedBox TrimBox ArtBox'.split()), PrintArea=checkPDFNames(*'MediaBox CropBox BleedBox TrimBox ArtBox'.split()), PrintClip=checkPDFNames(*'MediaBox CropBox BleedBox TrimBox ArtBox'.split()), PrintScaling=checkPDFNames(*'None AppDefault'.split()), Duplex=checkPDFNames(*'Simplex DuplexFlipShortEdge DuplexFlipLongEdge'.split()))