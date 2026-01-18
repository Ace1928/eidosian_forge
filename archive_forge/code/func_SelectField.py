from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def SelectField(title, value, options, xmin, ymin, xmax, ymax, page, font='Helvetica-Bold', fontsize=9, R=0, G=0, B=0.627):
    from reportlab.pdfbase.pdfdoc import PDFString, PDFName, PDFArray
    if value not in options:
        raise ValueError('value %s must be one of options %s' % (repr(value), repr(options)))
    fontname = FORMFONTNAMES[font]
    optionstrings = list(map(PDFString, options))
    optionarray = PDFArray(optionstrings)
    return PDFPattern(SelectFieldPattern, Options=optionarray, Selected=PDFString(value), Page=page, Name=PDFString(title), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, fontname=PDFName(fontname), fontsize=fontsize, R=R, G=G, B=B)