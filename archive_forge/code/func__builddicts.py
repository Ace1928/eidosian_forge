from fontTools.misc.textTools import tostr
import re
def _builddicts():
    import re
    lines = _aglText.splitlines()
    parseAGL_RE = re.compile('([A-Za-z0-9]+);((?:[0-9A-F]{4})(?: (?:[0-9A-F]{4}))*)$')
    for line in lines:
        if not line or line[:1] == '#':
            continue
        m = parseAGL_RE.match(line)
        if not m:
            raise AGLError('syntax error in glyphlist.txt: %s' % repr(line[:20]))
        unicodes = m.group(2)
        assert len(unicodes) % 5 == 4
        unicodes = [int(unicode, 16) for unicode in unicodes.split()]
        glyphName = tostr(m.group(1))
        LEGACY_AGL2UV[glyphName] = unicodes
    lines = _aglfnText.splitlines()
    parseAGLFN_RE = re.compile('([0-9A-F]{4});([A-Za-z0-9]+);.*?$')
    for line in lines:
        if not line or line[:1] == '#':
            continue
        m = parseAGLFN_RE.match(line)
        if not m:
            raise AGLError('syntax error in aglfn.txt: %s' % repr(line[:20]))
        unicode = m.group(1)
        assert len(unicode) == 4
        unicode = int(unicode, 16)
        glyphName = tostr(m.group(2))
        AGL2UV[glyphName] = unicode
        UV2AGL[unicode] = glyphName