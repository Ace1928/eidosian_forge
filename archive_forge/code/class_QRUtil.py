import re
import itertools
class QRUtil:
    PATTERN_POSITION_TABLE = [[], [6, 18], [6, 22], [6, 26], [6, 30], [6, 34], [6, 22, 38], [6, 24, 42], [6, 26, 46], [6, 28, 50], [6, 30, 54], [6, 32, 58], [6, 34, 62], [6, 26, 46, 66], [6, 26, 48, 70], [6, 26, 50, 74], [6, 30, 54, 78], [6, 30, 56, 82], [6, 30, 58, 86], [6, 34, 62, 90], [6, 28, 50, 72, 94], [6, 26, 50, 74, 98], [6, 30, 54, 78, 102], [6, 28, 54, 80, 106], [6, 32, 58, 84, 110], [6, 30, 58, 86, 114], [6, 34, 62, 90, 118], [6, 26, 50, 74, 98, 122], [6, 30, 54, 78, 102, 126], [6, 26, 52, 78, 104, 130], [6, 30, 56, 82, 108, 134], [6, 34, 60, 86, 112, 138], [6, 30, 58, 86, 114, 142], [6, 34, 62, 90, 118, 146], [6, 30, 54, 78, 102, 126, 150], [6, 24, 50, 76, 102, 128, 154], [6, 28, 54, 80, 106, 132, 158], [6, 32, 58, 84, 110, 136, 162], [6, 26, 54, 82, 110, 138, 166], [6, 30, 58, 86, 114, 142, 170]]
    G15 = 1 << 10 | 1 << 8 | 1 << 5 | 1 << 4 | 1 << 2 | 1 << 1 | 1 << 0
    G18 = 1 << 12 | 1 << 11 | 1 << 10 | 1 << 9 | 1 << 8 | 1 << 5 | 1 << 2 | 1 << 0
    G15_MASK = 1 << 14 | 1 << 12 | 1 << 10 | 1 << 4 | 1 << 1

    @staticmethod
    def getBCHTypeInfo(data):
        d = data << 10
        while QRUtil.getBCHDigit(d) - QRUtil.getBCHDigit(QRUtil.G15) >= 0:
            d ^= QRUtil.G15 << QRUtil.getBCHDigit(d) - QRUtil.getBCHDigit(QRUtil.G15)
        return (data << 10 | d) ^ QRUtil.G15_MASK

    @staticmethod
    def getBCHTypeNumber(data):
        d = data << 12
        while QRUtil.getBCHDigit(d) - QRUtil.getBCHDigit(QRUtil.G18) >= 0:
            d ^= QRUtil.G18 << QRUtil.getBCHDigit(d) - QRUtil.getBCHDigit(QRUtil.G18)
        return data << 12 | d

    @staticmethod
    def getBCHDigit(data):
        digit = 0
        while data != 0:
            digit += 1
            data >>= 1
        return digit

    @staticmethod
    def getPatternPosition(version):
        return QRUtil.PATTERN_POSITION_TABLE[version - 1]
    maskPattern = {0: lambda i, j: (i + j) % 2 == 0, 1: lambda i, j: i % 2 == 0, 2: lambda i, j: j % 3 == 0, 3: lambda i, j: (i + j) % 3 == 0, 4: lambda i, j: (i // 2 + j // 3) % 2 == 0, 5: lambda i, j: i * j % 2 + i * j % 3 == 0, 6: lambda i, j: (i * j % 2 + i * j % 3) % 2 == 0, 7: lambda i, j: (i * j % 3 + (i + j) % 2) % 2 == 0}

    @classmethod
    def getMask(cls, maskPattern):
        return cls.maskPattern[maskPattern]

    @staticmethod
    def getErrorCorrectPolynomial(errorCorrectLength):
        a = QRPolynomial([1], 0)
        for i in range(errorCorrectLength):
            a = a.multiply(QRPolynomial([1, QRMath.gexp(i)], 0))
        return a

    @classmethod
    def maskScoreRule1vert(cls, modules):
        score = 0
        lastCount = [0]
        lastRow = None
        for row in modules:
            if lastRow:
                changed = [a ^ b for a, b in zip(row, lastRow)]
                scores = [a and b - 4 + 3 for a, b in zip_longest(changed, lastCount, fillvalue=0) if b >= 4]
                score += sum(scores)
                lastCount = [0 if a else b + 1 for a, b in zip_longest(changed, lastCount, fillvalue=0)]
            lastRow = row
        score += sum([b - 4 + 3 for b in lastCount if b >= 4])
        return score

    @classmethod
    def maskScoreRule2(cls, modules):
        score = 0
        lastRow = modules[0]
        for row in modules[1:]:
            lastCol0, lastCol1 = (row[0], lastRow[0])
            for col0, col1 in zip(row[1:], lastRow[1:]):
                if col0 == col1 == lastCol0 == lastCol1:
                    score += 3
                lastCol0, lastCol1 = (col0, col1)
            lastRow = row
        return score

    @classmethod
    def maskScoreRule3hor(cls, modules, pattern=[True, False, True, True, True, False, True, False, False, False, False]):
        patternlen = len(pattern)
        score = 0
        for row in modules:
            j = 0
            maxj = len(row) - patternlen
            while j < maxj:
                if row[j:j + patternlen] == pattern:
                    score += 40
                    j += patternlen
                else:
                    j += 1
        return score

    @classmethod
    def maskScoreRule4(cls, modules):
        cellCount = len(modules) ** 2
        count = sum((sum(row) for row in modules))
        return 10 * (abs(100 * count // cellCount - 50) // 5)

    @classmethod
    def getLostPoint(cls, qrCode):
        lostPoint = 0
        lostPoint += cls.maskScoreRule1vert(qrCode.modules)
        lostPoint += cls.maskScoreRule1vert(zip(*qrCode.modules))
        lostPoint += cls.maskScoreRule2(qrCode.modules)
        lostPoint += cls.maskScoreRule3hor(qrCode.modules)
        lostPoint += cls.maskScoreRule3hor(zip(*qrCode.modules))
        lostPoint += cls.maskScoreRule4(qrCode.modules)
        return lostPoint