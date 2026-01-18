import re
import itertools
class QRCode:

    def __init__(self, version, errorCorrectLevel):
        self.version = version
        self.errorCorrectLevel = errorCorrectLevel
        self.modules = None
        self.moduleCount = 0
        self.dataCache = None
        self.dataList = []

    def addData(self, data):
        if isinstance(data, QR):
            newData = data
        else:
            for conv in (QRNumber, QRAlphaNum, QRKanji, QR8bitByte):
                try:
                    newData = conv(data)
                    break
                except ValueError:
                    pass
            else:
                raise ValueError
        self.dataList.append(newData)
        self.dataCache = None

    def isDark(self, row, col):
        return self.modules[row][col]

    def getModuleCount(self):
        return self.moduleCount

    def calculate_version(self):
        for version in range(1, 40):
            rsBlocks = QRRSBlock.getRSBlocks(version, self.errorCorrectLevel)
            totalDataCount = sum((block.dataCount for block in rsBlocks))
            length = 0
            for data in self.dataList:
                length += 4
                length += data.getLengthBits(version)
                length += data.bitlength
            if length <= totalDataCount * 8:
                break
        return version

    def make(self):
        if self.version is None:
            self.version = self.calculate_version()
        self.makeImpl(False, self.getBestMaskPattern())

    def makeImpl(self, test, maskPattern):
        self.moduleCount = self.version * 4 + 17
        self.modules = [[False] * self.moduleCount for x in range(self.moduleCount)]
        self.setupPositionProbePattern(0, 0)
        self.setupPositionProbePattern(self.moduleCount - 7, 0)
        self.setupPositionProbePattern(0, self.moduleCount - 7)
        self.setupPositionAdjustPattern()
        self.setupTimingPattern()
        self.setupTypeInfo(test, maskPattern)
        if self.version >= 7:
            self.setupTypeNumber(test)
        if self.dataCache == None:
            self.dataCache = QRCode.createData(self.version, self.errorCorrectLevel, self.dataList)
        self.mapData(self.dataCache, maskPattern)
    _positionProbePattern = [[True, True, True, True, True, True, True], [True, False, False, False, False, False, True], [True, False, True, True, True, False, True], [True, False, True, True, True, False, True], [True, False, True, True, True, False, True], [True, False, False, False, False, False, True], [True, True, True, True, True, True, True]]

    def setupPositionProbePattern(self, row, col):
        if row == 0:
            self.modules[row + 7][col:col + 7] = [False] * 7
            if col == 0:
                self.modules[row + 7][col + 7] = False
            else:
                self.modules[row + 7][col - 1] = False
        else:
            self.modules[row - 1][col:col + 8] = [False] * 8
        for r, data in enumerate(self._positionProbePattern):
            self.modules[row + r][col:col + 7] = data
            if col == 0:
                self.modules[row + r][col + 7] = False
            else:
                self.modules[row + r][col - 1] = False

    def getBestMaskPattern(self):
        minLostPoint = 0
        pattern = 0
        for i in range(8):
            self.makeImpl(True, i)
            lostPoint = QRUtil.getLostPoint(self)
            if i == 0 or minLostPoint > lostPoint:
                minLostPoint = lostPoint
                pattern = i
        return pattern

    def setupTimingPattern(self):
        for r in range(8, self.moduleCount - 8):
            self.modules[r][6] = r % 2 == 0
        self.modules[6][8:self.moduleCount - 8] = itertools.islice(itertools.cycle([True, False]), self.moduleCount - 16)
    _positionAdjustPattern = [[True, True, True, True, True], [True, False, False, False, True], [True, False, True, False, True], [True, False, False, False, True], [True, True, True, True, True]]

    def setupPositionAdjustPattern(self):
        pos = QRUtil.getPatternPosition(self.version)
        maxpos = self.moduleCount - 8
        for row, col in itertools.product(pos, pos):
            if col <= 8 and (row <= 8 or row >= maxpos):
                continue
            elif col >= maxpos and row <= 8:
                continue
            for r, data in enumerate(self._positionAdjustPattern):
                self.modules[row + r - 2][col - 2:col + 3] = data

    def setupTypeNumber(self, test):
        bits = QRUtil.getBCHTypeNumber(self.version)
        for i in range(18):
            mod = not test and bits >> i & 1 == 1
            self.modules[i // 3][i % 3 + self.moduleCount - 8 - 3] = mod
        for i in range(18):
            mod = not test and bits >> i & 1 == 1
            self.modules[i % 3 + self.moduleCount - 8 - 3][i // 3] = mod

    def setupTypeInfo(self, test, maskPattern):
        data = self.errorCorrectLevel << 3 | maskPattern
        bits = QRUtil.getBCHTypeInfo(data)
        for i in range(15):
            mod = not test and bits >> i & 1 == 1
            if i < 6:
                self.modules[i][8] = mod
            elif i < 8:
                self.modules[i + 1][8] = mod
            else:
                self.modules[self.moduleCount - 15 + i][8] = mod
        for i in range(15):
            mod = not test and bits >> i & 1 == 1
            if i < 8:
                self.modules[8][self.moduleCount - i - 1] = mod
            elif i < 9:
                self.modules[8][15 - i - 1 + 1] = mod
            else:
                self.modules[8][15 - i - 1] = mod
        self.modules[self.moduleCount - 8][8] = not test

    def _dataPosIterator(self):
        cols = itertools.chain(range(self.moduleCount - 1, 6, -2), range(5, 0, -2))
        rows = (list(range(9, self.moduleCount - 8)), list(itertools.chain(range(6), range(7, self.moduleCount))), list(range(9, self.moduleCount)))
        rrows = tuple((list(reversed(r)) for r in rows))
        ppos = QRUtil.getPatternPosition(self.version)
        ppos = set(itertools.chain.from_iterable(((p - 2, p - 1, p, p + 1, p + 2) for p in ppos)))
        maxpos = self.moduleCount - 11
        for col in cols:
            rows, rrows = (rrows, rows)
            if col <= 8:
                rowidx = 0
            elif col >= self.moduleCount - 8:
                rowidx = 2
            else:
                rowidx = 1
            for row in rows[rowidx]:
                for c in range(2):
                    c = col - c
                    if self.version >= 7:
                        if row < 6 and c >= self.moduleCount - 11:
                            continue
                        elif col < 6 and row >= self.moduleCount - 11:
                            continue
                    if row in ppos and c in ppos:
                        if not (row < 11 and (c < 11 or c > maxpos) or (c < 11 and (row < 11 or row > maxpos))):
                            continue
                    yield (c, row)
    _dataPosList = None

    def dataPosIterator(self):
        if not self._dataPosList:
            self._dataPosList = list(self._dataPosIterator())
        return self._dataPosList

    def _dataBitIterator(self, data):
        for byte in data:
            for bit in [128, 64, 32, 16, 8, 4, 2, 1]:
                yield bool(byte & bit)
    _dataBitList = None

    def dataBitIterator(self, data):
        if not self._dataBitList:
            self._dataBitList = list(self._dataBitIterator(data))
        return iter(self._dataBitList)

    def mapData(self, data, maskPattern):
        bits = self.dataBitIterator(data)
        mask = QRUtil.getMask(maskPattern)
        for (col, row), dark in zip_longest(self.dataPosIterator(), bits, fillvalue=False):
            self.modules[row][col] = dark ^ mask(row, col)
    PAD0 = 236
    PAD1 = 17

    @staticmethod
    def createData(version, errorCorrectLevel, dataList):
        rsBlocks = QRRSBlock.getRSBlocks(version, errorCorrectLevel)
        buffer = QRBitBuffer()
        for data in dataList:
            data.write(buffer, version)
        totalDataCount = 0
        for block in rsBlocks:
            totalDataCount += block.dataCount
        if buffer.getLengthInBits() > totalDataCount * 8:
            raise Exception('code length overflow. (%d > %d)' % (buffer.getLengthInBits(), totalDataCount * 8))
        if buffer.getLengthInBits() + 4 <= totalDataCount * 8:
            buffer.put(0, 4)
        while buffer.getLengthInBits() % 8 != 0:
            buffer.putBit(False)
        while True:
            if buffer.getLengthInBits() >= totalDataCount * 8:
                break
            buffer.put(QRCode.PAD0, 8)
            if buffer.getLengthInBits() >= totalDataCount * 8:
                break
            buffer.put(QRCode.PAD1, 8)
        return QRCode.createBytes(buffer, rsBlocks)

    @staticmethod
    def createBytes(buffer, rsBlocks):
        offset = 0
        maxDcCount = 0
        maxEcCount = 0
        totalCodeCount = 0
        dcdata = []
        ecdata = []
        for block in rsBlocks:
            totalCodeCount += block.totalCount
            dcCount = block.dataCount
            ecCount = block.totalCount - dcCount
            maxDcCount = max(maxDcCount, dcCount)
            maxEcCount = max(maxEcCount, ecCount)
            dcdata.append(buffer.buffer[offset:offset + dcCount])
            offset += dcCount
            rsPoly = QRUtil.getErrorCorrectPolynomial(ecCount)
            rawPoly = QRPolynomial(dcdata[-1], rsPoly.getLength() - 1)
            modPoly = rawPoly.mod(rsPoly)
            rLen = rsPoly.getLength() - 1
            mLen = modPoly.getLength()
            ecdata.append([modPoly.get(i) if i >= 0 else 0 for i in range(mLen - rLen, mLen)])
        data = [d for dd in itertools.chain(zip_longest(*dcdata), zip_longest(*ecdata)) for d in dd if d is not None]
        return data