import math
import warnings
from Bio import SeqUtils, Seq
from Bio import BiopythonWarning
def Tm_NN(seq, check=True, strict=True, c_seq=None, shift=0, nn_table=None, tmm_table=None, imm_table=None, de_table=None, dnac1=25, dnac2=25, selfcomp=False, Na=50, K=0, Tris=0, Mg=0, dNTPs=0, saltcorr=5):
    """Return the Tm using nearest neighbor thermodynamics.

    Arguments:
     - seq: The primer/probe sequence as string or Biopython sequence object.
       For RNA/DNA hybridizations seq must be the RNA sequence.
     - c_seq: Complementary sequence. The sequence of the template/target in
       3'->5' direction. c_seq is necessary for mismatch correction and
       dangling-ends correction. Both corrections will automatically be
       applied if mismatches or dangling ends are present. Default=None.
     - shift: Shift of the primer/probe sequence on the template/target
       sequence, e.g.::

                           shift=0       shift=1        shift= -1
        Primer (seq):      5' ATGC...    5'  ATGC...    5' ATGC...
        Template (c_seq):  3' TACG...    3' CTACG...    3'  ACG...

       The shift parameter is necessary to align seq and c_seq if they have
       different lengths or if they should have dangling ends. Default=0
     - table: Thermodynamic NN values, eight tables are implemented:
       For DNA/DNA hybridizations:

        - DNA_NN1: values from Breslauer et al. (1986)
        - DNA_NN2: values from Sugimoto et al. (1996)
        - DNA_NN3: values from Allawi & SantaLucia (1997) (default)
        - DNA_NN4: values from SantaLucia & Hicks (2004)

       For RNA/RNA hybridizations:

        - RNA_NN1: values from Freier et al. (1986)
        - RNA_NN2: values from Xia et al. (1998)
        - RNA_NN3: values from Chen et al. (2012)

       For RNA/DNA hybridizations:

        - R_DNA_NN1: values from Sugimoto et al. (1995)
          Note that ``seq`` must be the RNA sequence.

       Use the module's maketable method to make a new table or to update one
       one of the implemented tables.
     - tmm_table: Thermodynamic values for terminal mismatches.
       Default: DNA_TMM1 (SantaLucia & Peyret, 2001)
     - imm_table: Thermodynamic values for internal mismatches, may include
       insosine mismatches. Default: DNA_IMM1 (Allawi & SantaLucia, 1997-1998;
       Peyret et al., 1999; Watkins & SantaLucia, 2005)
     - de_table: Thermodynamic values for dangling ends:

        - DNA_DE1: for DNA. Values from Bommarito et al. (2000) (default)
        - RNA_DE1: for RNA. Values from Turner & Mathews (2010)

     - dnac1: Concentration of the higher concentrated strand [nM]. Typically
       this will be the primer (for PCR) or the probe. Default=25.
     - dnac2: Concentration of the lower concentrated strand [nM]. In PCR this
       is the template strand which concentration is typically very low and may
       be ignored (dnac2=0). In oligo/oligo hybridization experiments, dnac1
       equals dnac1. Default=25.
       MELTING and Primer3Plus use k = [Oligo(Total)]/4 by default. To mimic
       this behaviour, you have to divide [Oligo(Total)] by 2 and assign this
       concentration to dnac1 and dnac2. E.g., Total oligo concentration of
       50 nM in Primer3Plus means dnac1=25, dnac2=25.
     - selfcomp: Is the sequence self-complementary? Default=False. If 'True'
       the primer is thought binding to itself, thus dnac2 is not considered.
     - Na, K, Tris, Mg, dNTPs: See method 'Tm_GC' for details. Defaults: Na=50,
       K=0, Tris=0, Mg=0, dNTPs=0.
     - saltcorr: See method 'Tm_GC'. Default=5. 0 means no salt correction.

    """
    if not nn_table:
        nn_table = DNA_NN3
    if not tmm_table:
        tmm_table = DNA_TMM1
    if not imm_table:
        imm_table = DNA_IMM1
    if not de_table:
        de_table = DNA_DE1
    seq = str(seq)
    if not c_seq:
        c_seq = Seq.Seq(seq).complement()
    c_seq = str(c_seq)
    if check:
        seq = _check(seq, 'Tm_NN')
        c_seq = _check(c_seq, 'Tm_NN')
    tmp_seq = seq
    tmp_cseq = c_seq
    delta_h = 0
    delta_s = 0
    d_h = 0
    d_s = 1
    if shift or len(seq) != len(c_seq):
        if shift > 0:
            tmp_seq = '.' * shift + seq
        if shift < 0:
            tmp_cseq = '.' * abs(shift) + c_seq
        if len(tmp_cseq) > len(tmp_seq):
            tmp_seq += (len(tmp_cseq) - len(tmp_seq)) * '.'
        if len(tmp_cseq) < len(tmp_seq):
            tmp_cseq += (len(tmp_seq) - len(tmp_cseq)) * '.'
        while tmp_seq.startswith('..') or tmp_cseq.startswith('..'):
            tmp_seq = tmp_seq[1:]
            tmp_cseq = tmp_cseq[1:]
        while tmp_seq.endswith('..') or tmp_cseq.endswith('..'):
            tmp_seq = tmp_seq[:-1]
            tmp_cseq = tmp_cseq[:-1]
        if tmp_seq.startswith('.') or tmp_cseq.startswith('.'):
            left_de = tmp_seq[:2] + '/' + tmp_cseq[:2]
            try:
                delta_h += de_table[left_de][d_h]
                delta_s += de_table[left_de][d_s]
            except KeyError:
                _key_error(left_de, strict)
            tmp_seq = tmp_seq[1:]
            tmp_cseq = tmp_cseq[1:]
        if tmp_seq.endswith('.') or tmp_cseq.endswith('.'):
            right_de = tmp_cseq[-2:][::-1] + '/' + tmp_seq[-2:][::-1]
            try:
                delta_h += de_table[right_de][d_h]
                delta_s += de_table[right_de][d_s]
            except KeyError:
                _key_error(right_de, strict)
            tmp_seq = tmp_seq[:-1]
            tmp_cseq = tmp_cseq[:-1]
    left_tmm = tmp_cseq[:2][::-1] + '/' + tmp_seq[:2][::-1]
    if left_tmm in tmm_table:
        delta_h += tmm_table[left_tmm][d_h]
        delta_s += tmm_table[left_tmm][d_s]
        tmp_seq = tmp_seq[1:]
        tmp_cseq = tmp_cseq[1:]
    right_tmm = tmp_seq[-2:] + '/' + tmp_cseq[-2:]
    if right_tmm in tmm_table:
        delta_h += tmm_table[right_tmm][d_h]
        delta_s += tmm_table[right_tmm][d_s]
        tmp_seq = tmp_seq[:-1]
        tmp_cseq = tmp_cseq[:-1]
    delta_h += nn_table['init'][d_h]
    delta_s += nn_table['init'][d_s]
    if SeqUtils.gc_fraction(seq, 'ignore') == 0:
        delta_h += nn_table['init_allA/T'][d_h]
        delta_s += nn_table['init_allA/T'][d_s]
    else:
        delta_h += nn_table['init_oneG/C'][d_h]
        delta_s += nn_table['init_oneG/C'][d_s]
    if seq.startswith('T'):
        delta_h += nn_table['init_5T/A'][d_h]
        delta_s += nn_table['init_5T/A'][d_s]
    if seq.endswith('A'):
        delta_h += nn_table['init_5T/A'][d_h]
        delta_s += nn_table['init_5T/A'][d_s]
    ends = seq[0] + seq[-1]
    AT = ends.count('A') + ends.count('T')
    GC = ends.count('G') + ends.count('C')
    delta_h += nn_table['init_A/T'][d_h] * AT
    delta_s += nn_table['init_A/T'][d_s] * AT
    delta_h += nn_table['init_G/C'][d_h] * GC
    delta_s += nn_table['init_G/C'][d_s] * GC
    for basenumber in range(len(tmp_seq) - 1):
        neighbors = tmp_seq[basenumber:basenumber + 2] + '/' + tmp_cseq[basenumber:basenumber + 2]
        if neighbors in imm_table:
            delta_h += imm_table[neighbors][d_h]
            delta_s += imm_table[neighbors][d_s]
        elif neighbors[::-1] in imm_table:
            delta_h += imm_table[neighbors[::-1]][d_h]
            delta_s += imm_table[neighbors[::-1]][d_s]
        elif neighbors in nn_table:
            delta_h += nn_table[neighbors][d_h]
            delta_s += nn_table[neighbors][d_s]
        elif neighbors[::-1] in nn_table:
            delta_h += nn_table[neighbors[::-1]][d_h]
            delta_s += nn_table[neighbors[::-1]][d_s]
        else:
            _key_error(neighbors, strict)
    k = (dnac1 - dnac2 / 2.0) * 1e-09
    if selfcomp:
        k = dnac1 * 1e-09
        delta_h += nn_table['sym'][d_h]
        delta_s += nn_table['sym'][d_s]
    R = 1.987
    if saltcorr:
        corr = salt_correction(Na=Na, K=K, Tris=Tris, Mg=Mg, dNTPs=dNTPs, method=saltcorr, seq=seq)
    if saltcorr == 5:
        delta_s += corr
    melting_temp = 1000 * delta_h / (delta_s + R * math.log(k)) - 273.15
    if saltcorr in (1, 2, 3, 4):
        melting_temp += corr
    if saltcorr in (6, 7):
        melting_temp = 1 / (1 / (melting_temp + 273.15) + corr) - 273.15
    return melting_temp