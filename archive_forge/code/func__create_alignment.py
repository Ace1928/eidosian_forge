import numpy as np
from Bio.Align import Alignment, Alignments
from Bio.Align import bigbed, psl
from Bio.Align.bigbed import AutoSQLTable, Field
from Bio.Seq import Seq, reverse_complement, UndefinedSequenceError
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, Location
from Bio.SeqIO.InsdcIO import _insdc_location_string
def _create_alignment(self, chunk):
    chromId, tStart, tEnd, rest = chunk
    words = rest.decode().split('\t')
    if len(words) != 22:
        raise ValueError('Unexpected number of fields (%d, expected 22)' % len(words))
    target_record = self.targets[chromId]
    tSize = int(words[16])
    if len(target_record) != tSize:
        raise ValueError('Unexpected chromosome size %d (expected %d)' % (tSize, len(target_record)))
    strand = words[2]
    qName = words[0]
    qSize = int(words[12])
    blockCount = int(words[6])
    blockSizes = [int(blockSize) for blockSize in words[7].rstrip(',').split(',')]
    tStarts = [int(start) for start in words[8].rstrip(',').split(',')]
    qStarts = [int(start) for start in words[13].rstrip(',').split(',')]
    if len(blockSizes) != blockCount:
        raise ValueError('Inconsistent number of blocks (%d found, expected %d)' % (len(blockSizes), blockCount))
    if len(qStarts) != blockCount:
        raise ValueError('Inconsistent number of query start positions (%d found, expected %d)' % (len(qStarts), blockCount))
    if len(tStarts) != blockCount:
        raise ValueError('Inconsistent number of target start positions (%d found, expected %d)' % (len(qStarts), blockCount))
    qStarts = np.array(qStarts)
    tStarts = np.array(tStarts)
    tBlockSizes = np.array(blockSizes)
    query_sequence = words[14]
    if query_sequence == '':
        query_sequence = Seq(None, length=qSize)
    else:
        query_sequence = Seq(query_sequence)
        if len(query_sequence) != qSize:
            raise ValueError('Inconsistent query sequence length (%d, expected %d)' % (len(query_sequence), qSize))
    query_record = SeqRecord(query_sequence, id=qName)
    cds = words[15]
    if cds and cds != 'n/a':
        location = Location.fromstring(cds)
        feature = SeqFeature(location, type='CDS')
        query_record.features.append(feature)
    seqType = words[21]
    if seqType == '0':
        qBlockSizes = tBlockSizes
    elif seqType == '1':
        query_record.annotations['molecule_type'] = 'DNA'
        qBlockSizes = tBlockSizes
    elif seqType == '2':
        query_record.annotations['molecule_type'] = 'protein'
        qBlockSizes = tBlockSizes // 3
    else:
        raise ValueError("Unexpected sequence type '%s'" % seqType)
    tStarts += tStart
    qStrand = words[11]
    if qStrand == '-' and strand == '-':
        tStart, tEnd = (tEnd, tStart)
        qStarts = qSize - qStarts - qBlockSizes
        tStarts = tSize - tStarts - tBlockSizes
        qStarts = qStarts[::-1]
        tStarts = tStarts[::-1]
        qBlockSizes = qBlockSizes[::-1]
        tBlockSizes = tBlockSizes[::-1]
    qPosition = qStarts[0]
    tPosition = tStarts[0]
    coordinates = [[tPosition, qPosition]]
    for tB, qB, tS, qS in zip(tBlockSizes, qBlockSizes, tStarts, qStarts):
        if tS != tPosition:
            coordinates.append([tS, qPosition])
            tPosition = tS
        if qS != qPosition:
            coordinates.append([tPosition, qS])
            qPosition = qS
        tPosition += tB
        qPosition += qB
        coordinates.append([tPosition, qPosition])
    coordinates = np.array(coordinates).transpose()
    qStart = int(words[9])
    qEnd = int(words[10])
    if strand == '-':
        if qStrand == '-':
            coordinates[0, :] = tSize - coordinates[0, :]
        else:
            qStart, qEnd = (qEnd, qStart)
            coordinates[1, :] = qSize - coordinates[1, :]
    if tStart != coordinates[0, 0]:
        raise ValueError('Inconsistent tStart found (%d, expected %d)' % (tStart, coordinates[0, 0]))
    if tEnd != coordinates[0, -1]:
        raise ValueError('Inconsistent tEnd found (%d, expected %d)' % (tEnd, coordinates[0, -1]))
    if qStart != coordinates[1, 0]:
        raise ValueError('Inconsistent qStart found (%d, expected %d)' % (qStart, coordinates[1, 0]))
    if qEnd != coordinates[1, -1]:
        raise ValueError('Inconsistent qEnd found (%d, expected %d)' % (qEnd, coordinates[1, -1]))
    records = [target_record, query_record]
    alignment = Alignment(records, coordinates)
    alignment.annotations = {}
    score = words[1]
    try:
        score = float(score)
    except ValueError:
        pass
    else:
        if score.is_integer():
            score = int(score)
    alignment.score = score
    alignment.thickStart = int(words[3])
    alignment.thickEnd = int(words[4])
    alignment.itemRgb = words[5]
    alignment.matches = int(words[17])
    alignment.misMatches = int(words[18])
    alignment.repMatches = int(words[19])
    alignment.nCount = int(words[20])
    return alignment