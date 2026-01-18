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
class PDFPageLabel(PDFCatalog):
    __Comment__ = None
    __RefOnly__ = 0
    __Defaults__ = {}
    __NoDefault__ = 'Type S P St'.split()
    __convertible__ = 'ARABIC ROMAN_UPPER ROMAN_LOWER LETTERS_UPPER LETTERS_LOWER'
    ARABIC = 'D'
    ROMAN_UPPER = 'R'
    ROMAN_LOWER = 'r'
    LETTERS_UPPER = 'A'
    LETTERS_LOWER = 'a'

    def __init__(self, style=None, start=None, prefix=None):
        """
        A PDFPageLabel changes the style of page numbering as displayed in a PDF
        viewer. PDF page labels have nothing to do with 'physical' page numbers
        printed on a canvas, but instead influence the 'logical' page numbers
        displayed by PDF viewers. However, when using roman numerals (i, ii,
        iii...) or page prefixes for appendecies (A.1, A.2...) on the physical
        pages PDF page labels are necessary to change the logical page numbers
        displayed by the PDF viewer to match up with the physical numbers. A
        PDFPageLabel changes the properties of numbering at the page on which it
        appears (see the class 'PDFPageLabels' for specifying where a PDFPageLabel
        is associated) and all subsequent pages, until a new PDFPageLabel is
        encountered.

        The arguments to this initialiser determine the properties of all
        subsequent page labels. 'style' determines the numberings style, arabic,
        roman, letters; 'start' specifies the starting number; and 'prefix' any
        prefix to be applied to the page numbers. All these arguments can be left
        out or set to None.

        * style:

            - None:                       No numbering, can be used to display the prefix only.
            - PDFPageLabel.ARABIC:        Use arabic numbers: 1, 2, 3, 4...
            - PDFPageLabel.ROMAN_UPPER:   Use upper case roman numerals: I, II, III...
            - PDFPageLabel.ROMAN_LOWER:   Use lower case roman numerals: i, ii, iii...
            - PDFPageLabel.LETTERS_UPPER: Use upper case letters: A, B, C, D...
            - PDFPageLabel.LETTERS_LOWER: Use lower case letters: a, b, c, d...

        * start:

            -   An integer specifying the starting number for this PDFPageLabel. This
                can be used when numbering style changes to reset the page number back
                to one, ie from roman to arabic, or from arabic to appendecies. Can be
                any positive integer or None. I'm not sure what the effect of
                specifying None is, probably that page numbering continues with the
                current sequence, I'd have to check the spec to clarify though.

        * prefix:

            -   A string which is prefixed to the page numbers. Can be used to display
                appendecies in the format: A.1, A.2, ..., B.1, B.2, ... where a
                PDFPageLabel is used to set the properties for the first page of each
                appendix to restart the page numbering at one and set the prefix to the
                appropriate letter for current appendix. The prefix can also be used to
                display text only, if the 'style' is set to None. This can be used to
                display strings such as 'Front', 'Back', or 'Cover' for the covers on
                books.

        """
        if style:
            if style.upper() in self.__convertible__:
                style = getattr(self, style.upper())
            self.S = PDFName(style)
        if start:
            self.St = PDFnumber(start)
        if prefix:
            self.P = PDFString(prefix)

    def __lt__(self, oth):
        if rl_config.errorOnDuplicatePageLabelPage:
            raise DuplicatePageLabelPage()
        return False