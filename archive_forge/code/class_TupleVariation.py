from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
class TupleVariation(object):

    def __init__(self, axes, coordinates):
        self.axes = axes.copy()
        self.coordinates = list(coordinates)

    def __repr__(self):
        axes = ','.join(sorted(['%s=%s' % (name, value) for name, value in self.axes.items()]))
        return '<TupleVariation %s %s>' % (axes, self.coordinates)

    def __eq__(self, other):
        return self.coordinates == other.coordinates and self.axes == other.axes

    def getUsedPoints(self):
        if None not in self.coordinates:
            return frozenset()
        used = frozenset([i for i, p in enumerate(self.coordinates) if p is not None])
        return used if used else None

    def hasImpact(self):
        """Returns True if this TupleVariation has any visible impact.

        If the result is False, the TupleVariation can be omitted from the font
        without making any visible difference.
        """
        return any((c is not None for c in self.coordinates))

    def toXML(self, writer, axisTags):
        writer.begintag('tuple')
        writer.newline()
        for axis in axisTags:
            value = self.axes.get(axis)
            if value is not None:
                minValue, value, maxValue = value
                defaultMinValue = min(value, 0.0)
                defaultMaxValue = max(value, 0.0)
                if minValue == defaultMinValue and maxValue == defaultMaxValue:
                    writer.simpletag('coord', axis=axis, value=fl2str(value, 14))
                else:
                    attrs = [('axis', axis), ('min', fl2str(minValue, 14)), ('value', fl2str(value, 14)), ('max', fl2str(maxValue, 14))]
                    writer.simpletag('coord', attrs)
                writer.newline()
        wrote_any_deltas = False
        for i, delta in enumerate(self.coordinates):
            if type(delta) == tuple and len(delta) == 2:
                writer.simpletag('delta', pt=i, x=delta[0], y=delta[1])
                writer.newline()
                wrote_any_deltas = True
            elif type(delta) == int:
                writer.simpletag('delta', cvt=i, value=delta)
                writer.newline()
                wrote_any_deltas = True
            elif delta is not None:
                log.error('bad delta format')
                writer.comment('bad delta #%d' % i)
                writer.newline()
                wrote_any_deltas = True
        if not wrote_any_deltas:
            writer.comment('no deltas')
            writer.newline()
        writer.endtag('tuple')
        writer.newline()

    def fromXML(self, name, attrs, _content):
        if name == 'coord':
            axis = attrs['axis']
            value = str2fl(attrs['value'], 14)
            defaultMinValue = min(value, 0.0)
            defaultMaxValue = max(value, 0.0)
            minValue = str2fl(attrs.get('min', defaultMinValue), 14)
            maxValue = str2fl(attrs.get('max', defaultMaxValue), 14)
            self.axes[axis] = (minValue, value, maxValue)
        elif name == 'delta':
            if 'pt' in attrs:
                point = safeEval(attrs['pt'])
                x = safeEval(attrs['x'])
                y = safeEval(attrs['y'])
                self.coordinates[point] = (x, y)
            elif 'cvt' in attrs:
                cvt = safeEval(attrs['cvt'])
                value = safeEval(attrs['value'])
                self.coordinates[cvt] = value
            else:
                log.warning('bad delta format: %s' % ', '.join(sorted(attrs.keys())))

    def compile(self, axisTags, sharedCoordIndices={}, pointData=None):
        assert set(self.axes.keys()) <= set(axisTags), ('Unknown axis tag found.', self.axes.keys(), axisTags)
        tupleData = []
        auxData = []
        if pointData is None:
            usedPoints = self.getUsedPoints()
            if usedPoints is None:
                return (b'', b'')
            pointData = self.compilePoints(usedPoints)
        coord = self.compileCoord(axisTags)
        flags = sharedCoordIndices.get(coord)
        if flags is None:
            flags = EMBEDDED_PEAK_TUPLE
            tupleData.append(coord)
        intermediateCoord = self.compileIntermediateCoord(axisTags)
        if intermediateCoord is not None:
            flags |= INTERMEDIATE_REGION
            tupleData.append(intermediateCoord)
        if pointData:
            flags |= PRIVATE_POINT_NUMBERS
            auxData.append(pointData)
        auxData.append(self.compileDeltas())
        auxData = b''.join(auxData)
        tupleData.insert(0, struct.pack('>HH', len(auxData), flags))
        return (b''.join(tupleData), auxData)

    def compileCoord(self, axisTags):
        result = []
        axes = self.axes
        for axis in axisTags:
            triple = axes.get(axis)
            if triple is None:
                result.append(b'\x00\x00')
            else:
                result.append(struct.pack('>h', fl2fi(triple[1], 14)))
        return b''.join(result)

    def compileIntermediateCoord(self, axisTags):
        needed = False
        for axis in axisTags:
            minValue, value, maxValue = self.axes.get(axis, (0.0, 0.0, 0.0))
            defaultMinValue = min(value, 0.0)
            defaultMaxValue = max(value, 0.0)
            if minValue != defaultMinValue or maxValue != defaultMaxValue:
                needed = True
                break
        if not needed:
            return None
        minCoords = []
        maxCoords = []
        for axis in axisTags:
            minValue, value, maxValue = self.axes.get(axis, (0.0, 0.0, 0.0))
            minCoords.append(struct.pack('>h', fl2fi(minValue, 14)))
            maxCoords.append(struct.pack('>h', fl2fi(maxValue, 14)))
        return b''.join(minCoords + maxCoords)

    @staticmethod
    def decompileCoord_(axisTags, data, offset):
        coord = {}
        pos = offset
        for axis in axisTags:
            coord[axis] = fi2fl(struct.unpack('>h', data[pos:pos + 2])[0], 14)
            pos += 2
        return (coord, pos)

    @staticmethod
    def compilePoints(points):
        if not points:
            return b'\x00'
        points = list(points)
        points.sort()
        numPoints = len(points)
        result = bytearray()
        if numPoints < 128:
            result.append(numPoints)
        else:
            result.append(numPoints >> 8 | 128)
            result.append(numPoints & 255)
        MAX_RUN_LENGTH = 127
        pos = 0
        lastValue = 0
        while pos < numPoints:
            runLength = 0
            headerPos = len(result)
            result.append(0)
            useByteEncoding = None
            while pos < numPoints and runLength <= MAX_RUN_LENGTH:
                curValue = points[pos]
                delta = curValue - lastValue
                if useByteEncoding is None:
                    useByteEncoding = 0 <= delta <= 255
                if useByteEncoding and (delta > 255 or delta < 0):
                    break
                if useByteEncoding:
                    result.append(delta)
                else:
                    result.append(delta >> 8)
                    result.append(delta & 255)
                lastValue = curValue
                pos += 1
                runLength += 1
            if useByteEncoding:
                result[headerPos] = runLength - 1
            else:
                result[headerPos] = runLength - 1 | POINTS_ARE_WORDS
        return result

    @staticmethod
    def decompilePoints_(numPoints, data, offset, tableTag):
        """(numPoints, data, offset, tableTag) --> ([point1, point2, ...], newOffset)"""
        assert tableTag in ('cvar', 'gvar')
        pos = offset
        numPointsInData = data[pos]
        pos += 1
        if numPointsInData & POINTS_ARE_WORDS != 0:
            numPointsInData = (numPointsInData & POINT_RUN_COUNT_MASK) << 8 | data[pos]
            pos += 1
        if numPointsInData == 0:
            return (range(numPoints), pos)
        result = []
        while len(result) < numPointsInData:
            runHeader = data[pos]
            pos += 1
            numPointsInRun = (runHeader & POINT_RUN_COUNT_MASK) + 1
            point = 0
            if runHeader & POINTS_ARE_WORDS != 0:
                points = array.array('H')
                pointsSize = numPointsInRun * 2
            else:
                points = array.array('B')
                pointsSize = numPointsInRun
            points.frombytes(data[pos:pos + pointsSize])
            if sys.byteorder != 'big':
                points.byteswap()
            assert len(points) == numPointsInRun
            pos += pointsSize
            result.extend(points)
        absolute = []
        current = 0
        for delta in result:
            current += delta
            absolute.append(current)
        result = absolute
        del absolute
        badPoints = {str(p) for p in result if p < 0 or p >= numPoints}
        if badPoints:
            log.warning("point %s out of range in '%s' table" % (','.join(sorted(badPoints)), tableTag))
        return (result, pos)

    def compileDeltas(self):
        deltaX = []
        deltaY = []
        if self.getCoordWidth() == 2:
            for c in self.coordinates:
                if c is None:
                    continue
                deltaX.append(c[0])
                deltaY.append(c[1])
        else:
            for c in self.coordinates:
                if c is None:
                    continue
                deltaX.append(c)
        bytearr = bytearray()
        self.compileDeltaValues_(deltaX, bytearr)
        self.compileDeltaValues_(deltaY, bytearr)
        return bytearr

    @staticmethod
    def compileDeltaValues_(deltas, bytearr=None):
        """[value1, value2, value3, ...] --> bytearray

        Emits a sequence of runs. Each run starts with a
        byte-sized header whose 6 least significant bits
        (header & 0x3F) indicate how many values are encoded
        in this run. The stored length is the actual length
        minus one; run lengths are thus in the range [1..64].
        If the header byte has its most significant bit (0x80)
        set, all values in this run are zero, and no data
        follows. Otherwise, the header byte is followed by
        ((header & 0x3F) + 1) signed values.  If (header &
        0x40) is clear, the delta values are stored as signed
        bytes; if (header & 0x40) is set, the delta values are
        signed 16-bit integers.
        """
        if bytearr is None:
            bytearr = bytearray()
        pos = 0
        numDeltas = len(deltas)
        while pos < numDeltas:
            value = deltas[pos]
            if value == 0:
                pos = TupleVariation.encodeDeltaRunAsZeroes_(deltas, pos, bytearr)
            elif -128 <= value <= 127:
                pos = TupleVariation.encodeDeltaRunAsBytes_(deltas, pos, bytearr)
            else:
                pos = TupleVariation.encodeDeltaRunAsWords_(deltas, pos, bytearr)
        return bytearr

    @staticmethod
    def encodeDeltaRunAsZeroes_(deltas, offset, bytearr):
        pos = offset
        numDeltas = len(deltas)
        while pos < numDeltas and deltas[pos] == 0:
            pos += 1
        runLength = pos - offset
        while runLength >= 64:
            bytearr.append(DELTAS_ARE_ZERO | 63)
            runLength -= 64
        if runLength:
            bytearr.append(DELTAS_ARE_ZERO | runLength - 1)
        return pos

    @staticmethod
    def encodeDeltaRunAsBytes_(deltas, offset, bytearr):
        pos = offset
        numDeltas = len(deltas)
        while pos < numDeltas:
            value = deltas[pos]
            if not -128 <= value <= 127:
                break
            if value == 0 and pos + 1 < numDeltas and (deltas[pos + 1] == 0):
                break
            pos += 1
        runLength = pos - offset
        while runLength >= 64:
            bytearr.append(63)
            bytearr.extend(array.array('b', deltas[offset:offset + 64]))
            offset += 64
            runLength -= 64
        if runLength:
            bytearr.append(runLength - 1)
            bytearr.extend(array.array('b', deltas[offset:pos]))
        return pos

    @staticmethod
    def encodeDeltaRunAsWords_(deltas, offset, bytearr):
        pos = offset
        numDeltas = len(deltas)
        while pos < numDeltas:
            value = deltas[pos]
            if value == 0:
                break
            if -128 <= value <= 127 and pos + 1 < numDeltas and (-128 <= deltas[pos + 1] <= 127):
                break
            pos += 1
        runLength = pos - offset
        while runLength >= 64:
            bytearr.append(DELTAS_ARE_WORDS | 63)
            a = array.array('h', deltas[offset:offset + 64])
            if sys.byteorder != 'big':
                a.byteswap()
            bytearr.extend(a)
            offset += 64
            runLength -= 64
        if runLength:
            bytearr.append(DELTAS_ARE_WORDS | runLength - 1)
            a = array.array('h', deltas[offset:pos])
            if sys.byteorder != 'big':
                a.byteswap()
            bytearr.extend(a)
        return pos

    @staticmethod
    def decompileDeltas_(numDeltas, data, offset):
        """(numDeltas, data, offset) --> ([delta, delta, ...], newOffset)"""
        result = []
        pos = offset
        while len(result) < numDeltas:
            runHeader = data[pos]
            pos += 1
            numDeltasInRun = (runHeader & DELTA_RUN_COUNT_MASK) + 1
            if runHeader & DELTAS_ARE_ZERO != 0:
                result.extend([0] * numDeltasInRun)
            else:
                if runHeader & DELTAS_ARE_WORDS != 0:
                    deltas = array.array('h')
                    deltasSize = numDeltasInRun * 2
                else:
                    deltas = array.array('b')
                    deltasSize = numDeltasInRun
                deltas.frombytes(data[pos:pos + deltasSize])
                if sys.byteorder != 'big':
                    deltas.byteswap()
                assert len(deltas) == numDeltasInRun
                pos += deltasSize
                result.extend(deltas)
        assert len(result) == numDeltas
        return (result, pos)

    @staticmethod
    def getTupleSize_(flags, axisCount):
        size = 4
        if flags & EMBEDDED_PEAK_TUPLE != 0:
            size += axisCount * 2
        if flags & INTERMEDIATE_REGION != 0:
            size += axisCount * 4
        return size

    def getCoordWidth(self):
        """Return 2 if coordinates are (x, y) as in gvar, 1 if single values
        as in cvar, or 0 if empty.
        """
        firstDelta = next((c for c in self.coordinates if c is not None), None)
        if firstDelta is None:
            return 0
        if type(firstDelta) in (int, float):
            return 1
        if type(firstDelta) is tuple and len(firstDelta) == 2:
            return 2
        raise TypeError('invalid type of delta; expected (int or float) number, or Tuple[number, number]: %r' % firstDelta)

    def scaleDeltas(self, scalar):
        if scalar == 1.0:
            return
        coordWidth = self.getCoordWidth()
        self.coordinates = [None if d is None else d * scalar if coordWidth == 1 else (d[0] * scalar, d[1] * scalar) for d in self.coordinates]

    def roundDeltas(self):
        coordWidth = self.getCoordWidth()
        self.coordinates = [None if d is None else otRound(d) if coordWidth == 1 else (otRound(d[0]), otRound(d[1])) for d in self.coordinates]

    def calcInferredDeltas(self, origCoords, endPts):
        from fontTools.varLib.iup import iup_delta
        if self.getCoordWidth() == 1:
            raise TypeError("Only 'gvar' TupleVariation can have inferred deltas")
        if None in self.coordinates:
            if len(self.coordinates) != len(origCoords):
                raise ValueError('Expected len(origCoords) == %d; found %d' % (len(self.coordinates), len(origCoords)))
            self.coordinates = iup_delta(self.coordinates, origCoords, endPts)

    def optimize(self, origCoords, endPts, tolerance=0.5, isComposite=False):
        from fontTools.varLib.iup import iup_delta_optimize
        if None in self.coordinates:
            return
        deltaOpt = iup_delta_optimize(self.coordinates, origCoords, endPts, tolerance=tolerance)
        if None in deltaOpt:
            if isComposite and all((d is None for d in deltaOpt)):
                deltaOpt = [(0, 0)] + [None] * (len(deltaOpt) - 1)
            varOpt = TupleVariation(self.axes, deltaOpt)
            axisTags = sorted(self.axes.keys())
            tupleData, auxData = self.compile(axisTags)
            unoptimizedLength = len(tupleData) + len(auxData)
            tupleData, auxData = varOpt.compile(axisTags)
            optimizedLength = len(tupleData) + len(auxData)
            if optimizedLength < unoptimizedLength:
                self.coordinates = varOpt.coordinates

    def __imul__(self, scalar):
        self.scaleDeltas(scalar)
        return self

    def __iadd__(self, other):
        if not isinstance(other, TupleVariation):
            return NotImplemented
        deltas1 = self.coordinates
        length = len(deltas1)
        deltas2 = other.coordinates
        if len(deltas2) != length:
            raise ValueError('cannot sum TupleVariation deltas with different lengths')
        if self.getCoordWidth() == 2:
            for i, d2 in zip(range(length), deltas2):
                d1 = deltas1[i]
                try:
                    deltas1[i] = (d1[0] + d2[0], d1[1] + d2[1])
                except TypeError:
                    raise ValueError('cannot sum gvar deltas with inferred points')
        else:
            for i, d2 in zip(range(length), deltas2):
                d1 = deltas1[i]
                if d1 is not None and d2 is not None:
                    deltas1[i] = d1 + d2
                elif d1 is None and d2 is not None:
                    deltas1[i] = d2
        return self