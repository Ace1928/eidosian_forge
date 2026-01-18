import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_dt(record, line):
    value = line[5:]
    uprline = value.upper()
    cols = value.rstrip().split()
    if 'CREATED' in uprline or 'LAST SEQUENCE UPDATE' in uprline or 'LAST ANNOTATION UPDATE' in uprline:
        uprcols = uprline.split()
        rel_index = -1
        for index in range(len(uprcols)):
            if 'REL.' in uprcols[index]:
                rel_index = index
        assert rel_index >= 0, f'Could not find Rel. in DT line: {line}'
        version_index = rel_index + 1
        str_version = cols[version_index].rstrip(',')
        if str_version == '':
            version = 0
        elif '.' in str_version:
            version = str_version
        else:
            version = int(str_version)
        date = cols[0]
        if 'CREATED' in uprline:
            record.created = (date, version)
        elif 'LAST SEQUENCE UPDATE' in uprline:
            record.sequence_update = (date, version)
        elif 'LAST ANNOTATION UPDATE' in uprline:
            record.annotation_update = (date, version)
        else:
            raise SwissProtParserError('Unrecognised DT (DaTe) line', line=line)
    elif 'INTEGRATED INTO' in uprline or 'SEQUENCE VERSION' in uprline or 'ENTRY VERSION' in uprline:
        try:
            version = 0
            for s in cols[-1].split('.'):
                if s.isdigit():
                    version = int(s)
        except ValueError:
            version = 0
        date = cols[0].rstrip(',')
        if 'INTEGRATED' in uprline:
            record.created = (date, version)
        elif 'SEQUENCE VERSION' in uprline:
            record.sequence_update = (date, version)
        elif 'ENTRY VERSION' in uprline:
            record.annotation_update = (date, version)
        else:
            raise SwissProtParserError('Unrecognised DT (DaTe) line', line=line)
    else:
        raise SwissProtParserError('Failed to parse DT (DaTe) line', line=line)