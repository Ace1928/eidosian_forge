from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def FormFontsDictionary():
    from reportlab.pdfbase.pdfdoc import PDFDictionary
    fontsdictionary = PDFDictionary()
    fontsdictionary.__RefOnly__ = 1
    for fullname, shortname in FORMFONTNAMES.items():
        fontsdictionary[shortname] = FormFont(fullname, shortname)
    fontsdictionary['ZaDb'] = PDFPattern(ZaDbPattern)
    return fontsdictionary