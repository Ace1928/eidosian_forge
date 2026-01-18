import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_ft(record, line):
    name = line[5:13].rstrip()
    if name:
        if line[13:21] == '        ':
            location = line[21:80].rstrip()
            try:
                isoform_id, location = location.split(':')
            except ValueError:
                isoform_id = None
            try:
                from_res, to_res = location.split('..')
            except ValueError:
                from_res = location
                to_res = ''
            qualifiers = {}
        else:
            from_res = line[14:20].lstrip()
            to_res = line[21:27].lstrip()
            isoform_id = None
            description = line[34:75].rstrip()
            qualifiers = {'description': description}
        from_res = Position.fromstring(from_res, -1)
        if to_res == '':
            to_res = from_res + 1
        else:
            to_res = Position.fromstring(to_res)
        location = SimpleLocation(from_res, to_res, ref=isoform_id)
        feature = FeatureTable(location=location, type=name, id=None, qualifiers=qualifiers)
        record.features.append(feature)
        return
    feature = record.features[-1]
    if line[5:34] == '                             ':
        description = line[34:75].rstrip()
        if description.startswith('/FTId='):
            feature.id = description[6:].rstrip('.')
            return
        old_description = feature.qualifiers['description']
        if old_description.endswith('-'):
            description = f'{old_description}{description}'
        else:
            description = f'{old_description} {description}'
        if feature.type in ('VARSPLIC', 'VAR_SEQ'):
            try:
                first_seq, second_seq = description.split(' -> ')
            except ValueError:
                pass
            else:
                extra_info = ''
                extra_info_pos = second_seq.find(' (')
                if extra_info_pos != -1:
                    extra_info = second_seq[extra_info_pos:]
                    second_seq = second_seq[:extra_info_pos]
                first_seq = first_seq.replace(' ', '')
                second_seq = second_seq.replace(' ', '')
                description = first_seq + ' -> ' + second_seq + extra_info
        feature.qualifiers['description'] = description
    else:
        value = line[21:].rstrip()
        match = re.match('^/([a-z_]+)=', value)
        if match:
            qualifier_type = match.group(1)
            value = value[len(match.group(0)):]
            if not value.startswith('"'):
                raise ValueError('Missing starting quote in feature')
            if qualifier_type == 'id':
                if not value.endswith('"'):
                    raise ValueError('Missing closing quote for id')
                feature.id = value[1:-1]
            else:
                if value.endswith('"'):
                    value = value[1:-1]
                else:
                    value = value[1:]
                if qualifier_type in feature.qualifiers:
                    raise ValueError(f'Feature qualifier {qualifier_type!r} already exists for feature')
                feature.qualifiers[qualifier_type] = value
            return
        keys = list(feature.qualifiers.keys())
        key = keys[-1]
        description = value.rstrip('"')
        old_description = feature.qualifiers[key]
        if key == 'evidence' or old_description.endswith('-'):
            description = f'{old_description}{description}'
        else:
            description = f'{old_description} {description}'
        if feature.type == 'VAR_SEQ':
            try:
                first_seq, second_seq = description.split(' -> ')
            except ValueError:
                pass
            else:
                extra_info = ''
                extra_info_pos = second_seq.find(' (')
                if extra_info_pos != -1:
                    extra_info = second_seq[extra_info_pos:]
                    second_seq = second_seq[:extra_info_pos]
                first_seq = first_seq.replace(' ', '')
                second_seq = second_seq.replace(' ', '')
                description = first_seq + ' -> ' + second_seq + extra_info
        feature.qualifiers[key] = description