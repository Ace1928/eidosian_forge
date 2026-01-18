import re
import markupsafe
def _latexconverter(fg, bg, bold, underline, inverse):
    """
    Return start and end markup given foreground/background/bold/underline.

    """
    if (fg, bg, bold, underline, inverse) == (None, None, False, False, False):
        return ('', '')
    starttag, endtag = ('', '')
    if inverse:
        fg, bg = (bg, fg)
    if isinstance(fg, int):
        starttag += '\\textcolor{' + _ANSI_COLORS[fg] + '}{'
        endtag = '}' + endtag
    elif fg:
        starttag += '\\def\\tcRGB{\\textcolor[RGB]}\\expandafter'
        starttag += '\\tcRGB\\expandafter{{\\detokenize{{{},{},{}}}}}{{'.format(*fg)
        endtag = '}' + endtag
    elif inverse:
        starttag += '\\textcolor{ansi-default-inverse-fg}{'
        endtag = '}' + endtag
    if isinstance(bg, int):
        starttag += '\\setlength{\\fboxsep}{0pt}'
        starttag += '\\colorbox{' + _ANSI_COLORS[bg] + '}{'
        endtag = '\\strut}' + endtag
    elif bg:
        starttag += '\\setlength{\\fboxsep}{0pt}'
        starttag += '\\def\\cbRGB{\\colorbox[RGB]}\\expandafter'
        starttag += '\\cbRGB\\expandafter{{\\detokenize{{{},{},{}}}}}{{'.format(*bg)
        endtag = '\\strut}' + endtag
    elif inverse:
        starttag += '\\setlength{\\fboxsep}{0pt}'
        starttag += '\\colorbox{ansi-default-inverse-bg}{'
        endtag = '\\strut}' + endtag
    if bold:
        starttag += '\\textbf{'
        endtag = '}' + endtag
    if underline:
        starttag += '\\underline{'
        endtag = '}' + endtag
    return (starttag, endtag)