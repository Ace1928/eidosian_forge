from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class cmap_format_2(CmapSubtable):

    def setIDDelta(self, subHeader):
        subHeader.idDelta = 0
        minGI = subHeader.glyphIndexArray[0]
        for gid in subHeader.glyphIndexArray:
            if gid != 0 and gid < minGI:
                minGI = gid
        if minGI > 1:
            if minGI > 32767:
                subHeader.idDelta = -(65536 - minGI) - 1
            else:
                subHeader.idDelta = minGI - 1
            idDelta = subHeader.idDelta
            for i in range(subHeader.entryCount):
                gid = subHeader.glyphIndexArray[i]
                if gid > 0:
                    subHeader.glyphIndexArray[i] = gid - idDelta

    def decompile(self, data, ttFont):
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert data is None and ttFont is None, 'Need both data and ttFont arguments'
        data = self.data
        subHeaderKeys = []
        maxSubHeaderindex = 0
        allKeys = array.array('H')
        allKeys.frombytes(data[:512])
        data = data[512:]
        if sys.byteorder != 'big':
            allKeys.byteswap()
        subHeaderKeys = [key // 8 for key in allKeys]
        maxSubHeaderindex = max(subHeaderKeys)
        subHeaderList = []
        pos = 0
        for i in range(maxSubHeaderindex + 1):
            subHeader = SubHeader()
            subHeader.firstCode, subHeader.entryCount, subHeader.idDelta, subHeader.idRangeOffset = struct.unpack(subHeaderFormat, data[pos:pos + 8])
            pos += 8
            giDataPos = pos + subHeader.idRangeOffset - 2
            giList = array.array('H')
            giList.frombytes(data[giDataPos:giDataPos + subHeader.entryCount * 2])
            if sys.byteorder != 'big':
                giList.byteswap()
            subHeader.glyphIndexArray = giList
            subHeaderList.append(subHeader)
        self.data = b''
        cmap = {}
        notdefGI = 0
        for firstByte in range(256):
            subHeadindex = subHeaderKeys[firstByte]
            subHeader = subHeaderList[subHeadindex]
            if subHeadindex == 0:
                if firstByte < subHeader.firstCode or firstByte >= subHeader.firstCode + subHeader.entryCount:
                    continue
                else:
                    charCode = firstByte
                    offsetIndex = firstByte - subHeader.firstCode
                    gi = subHeader.glyphIndexArray[offsetIndex]
                    if gi != 0:
                        gi = (gi + subHeader.idDelta) % 65536
                    else:
                        continue
                cmap[charCode] = gi
            elif subHeader.entryCount:
                charCodeOffset = firstByte * 256 + subHeader.firstCode
                for offsetIndex in range(subHeader.entryCount):
                    charCode = charCodeOffset + offsetIndex
                    gi = subHeader.glyphIndexArray[offsetIndex]
                    if gi != 0:
                        gi = (gi + subHeader.idDelta) % 65536
                    else:
                        continue
                    cmap[charCode] = gi
        gids = list(cmap.values())
        charCodes = list(cmap.keys())
        self.cmap = _make_map(self.ttFont, charCodes, gids)

    def compile(self, ttFont):
        if self.data:
            return struct.pack('>HHH', self.format, self.length, self.language) + self.data
        kEmptyTwoCharCodeRange = -1
        notdefGI = 0
        items = sorted(self.cmap.items())
        charCodes = [item[0] for item in items]
        names = [item[1] for item in items]
        nameMap = ttFont.getReverseGlyphMap()
        try:
            gids = [nameMap[name] for name in names]
        except KeyError:
            nameMap = ttFont.getReverseGlyphMap(rebuild=True)
            try:
                gids = [nameMap[name] for name in names]
            except KeyError:
                gids = []
                for name in names:
                    try:
                        gid = nameMap[name]
                    except KeyError:
                        try:
                            if name[:3] == 'gid':
                                gid = int(name[3:])
                            else:
                                gid = ttFont.getGlyphID(name)
                        except:
                            raise KeyError(name)
                    gids.append(gid)
        subHeaderKeys = [kEmptyTwoCharCodeRange for x in range(256)]
        subHeaderList = []
        charCode = charCodes[0]
        if charCode > 255:
            subHeader = SubHeader()
            subHeader.firstCode = 0
            subHeader.entryCount = 0
            subHeader.idDelta = 0
            subHeader.idRangeOffset = 0
            subHeaderList.append(subHeader)
        lastFirstByte = -1
        items = zip(charCodes, gids)
        for charCode, gid in items:
            if gid == 0:
                continue
            firstbyte = charCode >> 8
            secondByte = charCode & 255
            if firstbyte != lastFirstByte:
                if lastFirstByte > -1:
                    self.setIDDelta(subHeader)
                    if lastFirstByte == 0:
                        for index in range(subHeader.entryCount):
                            charCode = subHeader.firstCode + index
                            subHeaderKeys[charCode] = 0
                    assert subHeader.entryCount == len(subHeader.glyphIndexArray), 'Error - subhead entry count does not match len of glyphID subrange.'
                subHeader = SubHeader()
                subHeader.firstCode = secondByte
                subHeader.entryCount = 1
                subHeader.glyphIndexArray.append(gid)
                subHeaderList.append(subHeader)
                subHeaderKeys[firstbyte] = len(subHeaderList) - 1
                lastFirstByte = firstbyte
            else:
                codeDiff = secondByte - (subHeader.firstCode + subHeader.entryCount)
                for i in range(codeDiff):
                    subHeader.glyphIndexArray.append(notdefGI)
                subHeader.glyphIndexArray.append(gid)
                subHeader.entryCount = subHeader.entryCount + codeDiff + 1
        self.setIDDelta(subHeader)
        subHeader = SubHeader()
        subHeader.firstCode = 0
        subHeader.entryCount = 0
        subHeader.idDelta = 0
        subHeader.idRangeOffset = 2
        subHeaderList.append(subHeader)
        emptySubheadIndex = len(subHeaderList) - 1
        for index in range(256):
            if subHeaderKeys[index] == kEmptyTwoCharCodeRange:
                subHeaderKeys[index] = emptySubheadIndex
        idRangeOffset = (len(subHeaderList) - 1) * 8 + 2
        subheadRangeLen = len(subHeaderList) - 1
        for index in range(subheadRangeLen):
            subHeader = subHeaderList[index]
            subHeader.idRangeOffset = 0
            for j in range(index):
                prevSubhead = subHeaderList[j]
                if prevSubhead.glyphIndexArray == subHeader.glyphIndexArray:
                    subHeader.idRangeOffset = prevSubhead.idRangeOffset - (index - j) * 8
                    subHeader.glyphIndexArray = []
                    break
            if subHeader.idRangeOffset == 0:
                subHeader.idRangeOffset = idRangeOffset
                idRangeOffset = idRangeOffset - 8 + subHeader.entryCount * 2
            else:
                idRangeOffset = idRangeOffset - 8
        length = 6 + 512 + 8 * len(subHeaderList)
        for subhead in subHeaderList[:-1]:
            length = length + len(subhead.glyphIndexArray) * 2
        dataList = [struct.pack('>HHH', 2, length, self.language)]
        for index in subHeaderKeys:
            dataList.append(struct.pack('>H', index * 8))
        for subhead in subHeaderList:
            dataList.append(struct.pack(subHeaderFormat, subhead.firstCode, subhead.entryCount, subhead.idDelta, subhead.idRangeOffset))
        for subhead in subHeaderList[:-1]:
            for gi in subhead.glyphIndexArray:
                dataList.append(struct.pack('>H', gi))
        data = bytesjoin(dataList)
        assert len(data) == length, 'Error: cmap format 2 is not same length as calculated! actual: ' + str(len(data)) + ' calc : ' + str(length)
        return data

    def fromXML(self, name, attrs, content, ttFont):
        self.language = safeEval(attrs['language'])
        if not hasattr(self, 'cmap'):
            self.cmap = {}
        cmap = self.cmap
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name != 'map':
                continue
            cmap[safeEval(attrs['code'])] = attrs['name']