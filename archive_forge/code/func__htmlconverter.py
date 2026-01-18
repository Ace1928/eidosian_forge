import re
import markupsafe
def _htmlconverter(fg, bg, bold, underline, inverse):
    """
    Return start and end tags for given foreground/background/bold/underline.

    """
    if (fg, bg, bold, underline, inverse) == (None, None, False, False, False):
        return ('', '')
    classes = []
    styles = []
    if inverse:
        fg, bg = (bg, fg)
    if isinstance(fg, int):
        classes.append(_ANSI_COLORS[fg] + '-fg')
    elif fg:
        styles.append('color: rgb({},{},{})'.format(*fg))
    elif inverse:
        classes.append('ansi-default-inverse-fg')
    if isinstance(bg, int):
        classes.append(_ANSI_COLORS[bg] + '-bg')
    elif bg:
        styles.append('background-color: rgb({},{},{})'.format(*bg))
    elif inverse:
        classes.append('ansi-default-inverse-bg')
    if bold:
        classes.append('ansi-bold')
    if underline:
        classes.append('ansi-underline')
    starttag = '<span'
    if classes:
        starttag += ' class="' + ' '.join(classes) + '"'
    if styles:
        starttag += ' style="' + '; '.join(styles) + '"'
    starttag += '>'
    return (starttag, '</span>')