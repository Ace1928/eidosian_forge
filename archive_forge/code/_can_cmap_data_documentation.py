from reportlab.pdfbase._cidfontdata import defaultUnicodeEncodings
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

This is a utility to 'can' the widths data for certain CID fonts.
Now we're using Unicode, we don't need 20 CMAP files for each Asian
language, nor the widths of the non-normal characters encoded in each
font.  we just want a dictionary of the character widths in a given
font which are NOT 1000 ems wide, keyed on Unicode character (not CID).

Running off CMAP files we get the following widths...::

    >>> font = UnicodeCIDFont('HeiseiMin-W3')
    >>> font.stringWidth(unicode(','), 10)
    2.5
    >>> font.stringWidth(unicode('m'), 10)
    7.7800000000000002
    >>> font.stringWidth(u'東京', 10)
    20.0
    >>> 

