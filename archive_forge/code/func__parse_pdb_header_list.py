import re
from Bio import File
def _parse_pdb_header_list(header):
    pdbh_dict = {'name': '', 'head': '', 'idcode': '', 'deposition_date': '1909-01-08', 'release_date': '1909-01-08', 'structure_method': 'unknown', 'resolution': None, 'structure_reference': 'unknown', 'journal_reference': 'unknown', 'author': '', 'compound': {'1': {'misc': ''}}, 'source': {'1': {'misc': ''}}, 'has_missing_residues': False, 'missing_residues': []}
    pdbh_dict['structure_reference'] = _get_references(header)
    pdbh_dict['journal_reference'] = _get_journal(header)
    comp_molid = '1'
    last_comp_key = 'misc'
    last_src_key = 'misc'
    for hh in header:
        h = re.sub('[\\s\\n\\r]*\\Z', '', hh)
        key = h[:6].strip()
        tail = h[10:].strip()
        if key == 'TITLE':
            name = _chop_end_codes(tail).lower()
            pdbh_dict['name'] = ' '.join([pdbh_dict['name'], name]).strip()
        elif key == 'HEADER':
            rr = re.search('\\d\\d-\\w\\w\\w-\\d\\d', tail)
            if rr is not None:
                pdbh_dict['deposition_date'] = _format_date(_nice_case(rr.group()))
            rr = re.search('\\s+([1-9][0-9A-Z]{3})\\s*\\Z', tail)
            if rr is not None:
                pdbh_dict['idcode'] = rr.group(1)
            head = _chop_end_misc(tail).lower()
            pdbh_dict['head'] = head
        elif key == 'COMPND':
            tt = re.sub('\\;\\s*\\Z', '', _chop_end_codes(tail)).lower()
            rec = re.search('\\d+\\.\\d+\\.\\d+\\.\\d+', tt)
            if rec:
                pdbh_dict['compound'][comp_molid]['ec_number'] = rec.group()
                tt = re.sub('\\((e\\.c\\.)*\\d+\\.\\d+\\.\\d+\\.\\d+\\)', '', tt)
            tok = tt.split(':')
            if len(tok) >= 2:
                ckey = tok[0]
                cval = re.sub('\\A\\s*', '', tok[1])
                if ckey == 'mol_id':
                    pdbh_dict['compound'][cval] = {'misc': ''}
                    comp_molid = cval
                    last_comp_key = 'misc'
                else:
                    pdbh_dict['compound'][comp_molid][ckey] = cval
                    last_comp_key = ckey
            else:
                pdbh_dict['compound'][comp_molid][last_comp_key] += tok[0] + ' '
        elif key == 'SOURCE':
            tt = re.sub('\\;\\s*\\Z', '', _chop_end_codes(tail)).lower()
            tok = tt.split(':')
            if len(tok) >= 2:
                ckey = tok[0]
                cval = re.sub('\\A\\s*', '', tok[1])
                if ckey == 'mol_id':
                    pdbh_dict['source'][cval] = {'misc': ''}
                    comp_molid = cval
                    last_src_key = 'misc'
                else:
                    pdbh_dict['source'][comp_molid][ckey] = cval
                    last_src_key = ckey
            else:
                pdbh_dict['source'][comp_molid][last_src_key] += tok[0] + ' '
        elif key == 'KEYWDS':
            kwd = _chop_end_codes(tail).lower()
            if 'keywords' in pdbh_dict:
                pdbh_dict['keywords'] += ' ' + kwd
            else:
                pdbh_dict['keywords'] = kwd
        elif key == 'EXPDTA':
            expd = _chop_end_codes(tail)
            expd = re.sub('\\s\\s\\s\\s\\s\\s\\s.*\\Z', '', expd)
            pdbh_dict['structure_method'] = expd.lower()
        elif key == 'CAVEAT':
            pass
        elif key == 'REVDAT':
            rr = re.search('\\d\\d-\\w\\w\\w-\\d\\d', tail)
            if rr is not None:
                pdbh_dict['release_date'] = _format_date(_nice_case(rr.group()))
        elif key == 'JRNL':
            if 'journal' in pdbh_dict:
                pdbh_dict['journal'] += tail
            else:
                pdbh_dict['journal'] = tail
        elif key == 'AUTHOR':
            auth = _nice_case(_chop_end_codes(tail))
            if 'author' in pdbh_dict:
                pdbh_dict['author'] += auth
            else:
                pdbh_dict['author'] = auth
        elif key == 'REMARK':
            if re.search('REMARK   2 RESOLUTION.', hh):
                r = _chop_end_codes(re.sub('REMARK   2 RESOLUTION.', '', hh))
                r = re.sub('\\s+ANGSTROM.*', '', r)
                try:
                    pdbh_dict['resolution'] = float(r)
                except ValueError:
                    pdbh_dict['resolution'] = None
            elif hh.startswith('REMARK 465'):
                if tail:
                    pdbh_dict['has_missing_residues'] = True
                    missing_res_info = _parse_remark_465(tail)
                    if missing_res_info:
                        pdbh_dict['missing_residues'].append(missing_res_info)
            elif hh.startswith('REMARK  99 ASTRAL'):
                if tail:
                    remark_99_keyval = tail.replace('ASTRAL ', '').split(': ')
                    if isinstance(remark_99_keyval, list) and len(remark_99_keyval) == 2:
                        if 'astral' not in pdbh_dict:
                            pdbh_dict['astral'] = {remark_99_keyval[0]: remark_99_keyval[1]}
                        else:
                            pdbh_dict['astral'][remark_99_keyval[0]] = remark_99_keyval[1]
        else:
            pass
    if pdbh_dict['structure_method'] == 'unknown':
        res = pdbh_dict['resolution']
        if res is not None and res > 0.0:
            pdbh_dict['structure_method'] = 'x-ray diffraction'
    return pdbh_dict