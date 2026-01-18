import copy
from collections.abc import Mapping, Iterable
from Bio import BiopythonWarning
from Bio import BiopythonExperimentalWarning
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.codonalign.codonseq import CodonSeq
from Bio.codonalign.codonalignment import CodonAlignment, mktest
import warnings
def _check_corr(pro, nucl, gap_char, codon_table, complete_protein=False, anchor_len=10):
    """Check if the nucleotide can be translated into the protein (PRIVATE).

    Expects two SeqRecord objects.
    """
    import re
    if not isinstance(pro, SeqRecord) or not isinstance(nucl, SeqRecord):
        raise TypeError('_check_corr accepts two SeqRecord object. Please check your input.')
    aa2re = _get_aa_regex(codon_table)
    pro_re = ''
    for aa in pro.seq:
        if aa != gap_char:
            pro_re += aa2re[aa]
    nucl_seq = str(nucl.seq.upper().replace(gap_char, ''))
    match = re.search(pro_re, nucl_seq)
    if match:
        return (match.span(), 0)
    else:
        pro_seq = str(pro.seq).replace(gap_char, '')
        anchors = [pro_seq[i:i + anchor_len] for i in range(0, len(pro_seq), anchor_len)]
        if len(anchors[-1]) < anchor_len:
            anchors[-1] = anchors[-2] + anchors[-1]
        pro_re = []
        anchor_distance = 0
        anchor_pos = []
        for i, anchor in enumerate(anchors):
            this_anchor_len = len(anchor)
            qcodon = ''
            fncodon = ''
            if this_anchor_len == anchor_len:
                for aa in anchor:
                    if complete_protein and i == 0:
                        qcodon += _codons2re(codon_table.start_codons)
                        fncodon += aa2re['X']
                        continue
                    qcodon += aa2re[aa]
                    fncodon += aa2re['X']
                match = re.search(qcodon, nucl_seq)
            elif this_anchor_len > anchor_len:
                last_qcodon = ''
                last_fcodon = ''
                for j in range(anchor_len, len(anchor)):
                    last_qcodon += aa2re[anchor[j]]
                    last_fcodon += aa2re['X']
                match = re.search(last_qcodon, nucl_seq)
            if match:
                anchor_pos.append((match.start(), match.end(), i))
                if this_anchor_len == anchor_len:
                    pro_re.append(qcodon)
                else:
                    pro_re.append(last_qcodon)
            elif this_anchor_len == anchor_len:
                pro_re.append(fncodon)
            else:
                pro_re.append(last_fcodon)
        full_pro_re = ''.join(pro_re)
        match = re.search(full_pro_re, nucl_seq)
        if match:
            return (match.span(), 1)
        else:
            first_anchor = True
            shift_id_pos = 0
            if first_anchor and anchor_pos[0][2] != 0:
                shift_val_lst = [1, 2, 3 * anchor_len - 2, 3 * anchor_len - 1, 0]
                sh_anc = anchors[0]
                for shift_val in shift_val_lst:
                    if shift_val == 0:
                        qcodon = None
                        break
                    if shift_val in (1, 2):
                        sh_nuc_len = anchor_len * 3 + shift_val
                    elif shift_val in (3 * anchor_len - 2, 3 * anchor_len - 1):
                        sh_nuc_len = anchor_len * 3 - (3 * anchor_len - shift_val)
                    if anchor_pos[0][0] >= sh_nuc_len:
                        sh_nuc = nucl_seq[anchor_pos[0][0] - sh_nuc_len:anchor_pos[0][0]]
                    else:
                        sh_nuc = nucl_seq[:anchor_pos[0][0]]
                    qcodon, shift_id_pos = _get_shift_anchor_re(sh_anc, sh_nuc, shift_val, aa2re, anchor_len, shift_id_pos)
                    if qcodon is not None and qcodon != -1:
                        pro_re[0] = qcodon
                        break
                if qcodon == -1:
                    warnings.warn(f'first frameshift detection failed for {nucl.id}', BiopythonWarning)
            for i in range(len(anchor_pos) - 1):
                shift_val = (anchor_pos[i + 1][0] - anchor_pos[i][0]) % (3 * anchor_len)
                sh_anc = ''.join(anchors[anchor_pos[i][2]:anchor_pos[i + 1][2]])
                sh_nuc = nucl_seq[anchor_pos[i][0]:anchor_pos[i + 1][0]]
                qcodon = None
                if shift_val != 0:
                    qcodon, shift_id_pos = _get_shift_anchor_re(sh_anc, sh_nuc, shift_val, aa2re, anchor_len, shift_id_pos)
                if qcodon is not None and qcodon != -1:
                    pro_re[anchor_pos[i][2]:anchor_pos[i + 1][2]] = [qcodon]
                    qcodon = None
                elif qcodon == -1:
                    warnings.warn(f'middle frameshift detection failed for {nucl.id}', BiopythonWarning)
            if anchor_pos[-1][2] + 1 == len(anchors) - 1:
                sh_anc = anchors[-1]
                this_anchor_len = len(sh_anc)
                shift_val_lst = [1, 2, 3 * this_anchor_len - 2, 3 * this_anchor_len - 1, 0]
                for shift_val in shift_val_lst:
                    if shift_val == 0:
                        qcodon = None
                        break
                    if shift_val in (1, 2):
                        sh_nuc_len = this_anchor_len * 3 + shift_val
                    elif shift_val in (3 * this_anchor_len - 2, 3 * this_anchor_len - 1):
                        sh_nuc_len = this_anchor_len * 3 - (3 * this_anchor_len - shift_val)
                    if len(nucl_seq) - anchor_pos[-1][0] >= sh_nuc_len:
                        sh_nuc = nucl_seq[anchor_pos[-1][0]:anchor_pos[-1][0] + sh_nuc_len]
                    else:
                        sh_nuc = nucl_seq[anchor_pos[-1][0]:]
                    qcodon, shift_id_pos = _get_shift_anchor_re(sh_anc, sh_nuc, shift_val, aa2re, this_anchor_len, shift_id_pos)
                    if qcodon is not None and qcodon != -1:
                        pro_re.pop()
                        pro_re[-1] = qcodon
                        break
                if qcodon == -1:
                    warnings.warn(f'last frameshift detection failed for {nucl.id}', BiopythonWarning)
            full_pro_re = ''.join(pro_re)
            match = re.search(full_pro_re, nucl_seq)
            if match:
                return (match.span(), 2, match)
            else:
                raise RuntimeError(f'Protein SeqRecord ({pro.id}) and Nucleotide SeqRecord ({nucl.id}) do not match!')