import math
import warnings
from Bio import SeqUtils, Seq
from Bio import BiopythonWarning
Return the Tm using nearest neighbor thermodynamics.

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

    