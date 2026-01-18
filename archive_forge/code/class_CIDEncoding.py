import os
import marshal
import time
from hashlib import md5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase._cidfontdata import allowedTypeFaces, allowedEncodings, CIDFontInfo, \
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfdoc
from reportlab.lib.rl_accel import escapePDF
from reportlab.rl_config import CMapSearchPath
from reportlab.lib.utils import isSeq, isBytes
class CIDEncoding(pdfmetrics.Encoding):
    """Multi-byte encoding.  These are loaded from CMAP files.

    A CMAP file is like a mini-codec.  It defines the correspondence
    between code points in the (multi-byte) input data and Character
    IDs. """

    def __init__(self, name, useCache=1):
        self.name = name
        self._mapFileHash = None
        self._codeSpaceRanges = []
        self._notDefRanges = []
        self._cmap = {}
        self.source = None
        if not DISABLE_CMAP:
            if useCache:
                from reportlab.lib.utils import get_rl_tempdir
                fontmapdir = get_rl_tempdir('FastCMAPS')
                if os.path.isfile(fontmapdir + os.sep + name + '.fastmap'):
                    self.fastLoad(fontmapdir)
                    self.source = fontmapdir + os.sep + name + '.fastmap'
                else:
                    self.parseCMAPFile(name)
                    self.source = 'CMAP: ' + name
                    self.fastSave(fontmapdir)
            else:
                self.parseCMAPFile(name)

    def _hash(self, text):
        hasher = md5()
        hasher.update(text)
        return hasher.digest()

    def parseCMAPFile(self, name):
        """This is a tricky one as CMAP files are Postscript
        ones.  Some refer to others with a 'usecmap'
        command"""
        cmapfile = findCMapFile(name)
        rawdata = open(cmapfile, 'r').read()
        self._mapFileHash = self._hash(rawdata)
        usecmap_pos = rawdata.find('usecmap')
        if usecmap_pos > -1:
            chunk = rawdata[0:usecmap_pos]
            words = chunk.split()
            otherCMAPName = words[-1]
            self.parseCMAPFile(otherCMAPName)
        words = rawdata.split()
        while words != []:
            if words[0] == 'begincodespacerange':
                words = words[1:]
                while words[0] != 'endcodespacerange':
                    strStart, strEnd, words = (words[0], words[1], words[2:])
                    start = int(strStart[1:-1], 16)
                    end = int(strEnd[1:-1], 16)
                    self._codeSpaceRanges.append((start, end))
            elif words[0] == 'beginnotdefrange':
                words = words[1:]
                while words[0] != 'endnotdefrange':
                    strStart, strEnd, strValue = words[0:3]
                    start = int(strStart[1:-1], 16)
                    end = int(strEnd[1:-1], 16)
                    value = int(strValue)
                    self._notDefRanges.append((start, end, value))
                    words = words[3:]
            elif words[0] == 'begincidrange':
                words = words[1:]
                while words[0] != 'endcidrange':
                    strStart, strEnd, strValue = words[0:3]
                    start = int(strStart[1:-1], 16)
                    end = int(strEnd[1:-1], 16)
                    value = int(strValue)
                    offset = 0
                    while start + offset <= end:
                        self._cmap[start + offset] = value + offset
                        offset = offset + 1
                    words = words[3:]
            else:
                words = words[1:]

    def translate(self, text):
        """Convert a string into a list of CIDs"""
        output = []
        cmap = self._cmap
        lastChar = ''
        for char in text:
            if lastChar != '':
                num = ord(lastChar) * 256 + ord(char)
            else:
                num = ord(char)
            lastChar = char
            found = 0
            for low, high in self._codeSpaceRanges:
                if low < num < high:
                    try:
                        cid = cmap[num]
                    except KeyError:
                        cid = 0
                        for low2, high2, notdef in self._notDefRanges:
                            if low2 < num < high2:
                                cid = notdef
                                break
                    output.append(cid)
                    found = 1
                    break
            if found:
                lastChar = ''
            else:
                lastChar = char
        return output

    def fastSave(self, directory):
        f = open(os.path.join(directory, self.name + '.fastmap'), 'wb')
        marshal.dump(self._mapFileHash, f)
        marshal.dump(self._codeSpaceRanges, f)
        marshal.dump(self._notDefRanges, f)
        marshal.dump(self._cmap, f)
        f.close()

    def fastLoad(self, directory):
        started = time.clock()
        f = open(os.path.join(directory, self.name + '.fastmap'), 'rb')
        self._mapFileHash = marshal.load(f)
        self._codeSpaceRanges = marshal.load(f)
        self._notDefRanges = marshal.load(f)
        self._cmap = marshal.load(f)
        f.close()
        finished = time.clock()

    def getData(self):
        """Simple persistence helper.  Return a dict with all that matters."""
        return {'mapFileHash': self._mapFileHash, 'codeSpaceRanges': self._codeSpaceRanges, 'notDefRanges': self._notDefRanges, 'cmap': self._cmap}