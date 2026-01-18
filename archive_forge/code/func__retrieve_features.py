from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def _retrieve_features(adaptor, primary_id):
    sql = 'SELECT seqfeature_id, type.name, "rank" FROM seqfeature join term type on (type_term_id = type.term_id) WHERE bioentry_id = %s ORDER BY "rank"'
    results = adaptor.execute_and_fetchall(sql, (primary_id,))
    seq_feature_list = []
    for seqfeature_id, seqfeature_type, seqfeature_rank in results:
        qvs = adaptor.execute_and_fetchall('SELECT name, value FROM seqfeature_qualifier_value  join term using (term_id) WHERE seqfeature_id = %s ORDER BY "rank"', (seqfeature_id,))
        qualifiers = {}
        for qv_name, qv_value in qvs:
            qualifiers.setdefault(qv_name, []).append(qv_value)
        qvs = adaptor.execute_and_fetchall('SELECT dbxref.dbname, dbxref.accession FROM dbxref join seqfeature_dbxref using (dbxref_id) WHERE seqfeature_dbxref.seqfeature_id = %s ORDER BY "rank"', (seqfeature_id,))
        for qv_name, qv_value in qvs:
            value = f'{qv_name}:{qv_value}'
            qualifiers.setdefault('db_xref', []).append(value)
        results = adaptor.execute_and_fetchall('SELECT location_id, start_pos, end_pos, strand FROM location WHERE seqfeature_id = %s ORDER BY "rank"', (seqfeature_id,))
        locations = []
        for location_id, start, end, strand in results:
            if start:
                start -= 1
            if strand == 0:
                strand = None
            if strand not in (+1, -1, None):
                raise ValueError('Invalid strand %s found in database for seqfeature_id %s' % (strand, seqfeature_id))
            if start is not None and end is not None and (end < start):
                import warnings
                from Bio import BiopythonWarning
                warnings.warn('Inverted location start/end (%i and %i) for seqfeature_id %s' % (start, end, seqfeature_id), BiopythonWarning)
            if start is None:
                start = SeqFeature.UnknownPosition()
            if end is None:
                end = SeqFeature.UnknownPosition()
            locations.append((location_id, start, end, strand))
        remote_results = adaptor.execute_and_fetchall('SELECT location_id, dbname, accession, version FROM location join dbxref using (dbxref_id) WHERE seqfeature_id = %s', (seqfeature_id,))
        lookup = {}
        for location_id, dbname, accession, version in remote_results:
            if version and version != '0':
                v = f'{accession}.{version}'
            else:
                v = accession
            if dbname == '':
                dbname = None
            lookup[location_id] = (dbname, v)
        feature = SeqFeature.SeqFeature(type=seqfeature_type)
        feature._seqfeature_id = seqfeature_id
        feature.qualifiers = qualifiers
        if len(locations) == 0:
            pass
        elif len(locations) == 1:
            location_id, start, end, strand = locations[0]
            feature.location_operator = _retrieve_location_qualifier_value(adaptor, location_id)
            dbname, version = lookup.get(location_id, (None, None))
            feature.location = SeqFeature.SimpleLocation(start, end)
            feature.location.strand = strand
            feature.location.ref_db = dbname
            feature.location.ref = version
        else:
            locs = []
            for location in locations:
                location_id, start, end, strand = location
                dbname, version = lookup.get(location_id, (None, None))
                locs.append(SeqFeature.SimpleLocation(start, end, strand=strand, ref=version, ref_db=dbname))
            strands = {_.strand for _ in locs}
            if len(strands) == 1 and -1 in strands:
                locs = locs[::-1]
            feature.location = SeqFeature.CompoundLocation(locs, 'join')
        seq_feature_list.append(feature)
    return seq_feature_list