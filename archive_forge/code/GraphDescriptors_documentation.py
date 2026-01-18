import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
 A topological index meant to quantify "complexity" of molecules.

     Consists of a sum of two terms, one representing the complexity
     of the bonding, the other representing the complexity of the
     distribution of heteroatoms.

     From S. H. Bertz, J. Am. Chem. Soc., vol 103, 3599-3601 (1981)

     "cutoff" is an integer value used to limit the computational
     expense.  A cutoff value tells the program to consider vertices
     topologically identical if their distance vectors (sets of
     distances to all other vertices) are equal out to the "cutoff"th
     nearest-neighbor.

     **NOTE**  The original implementation had the following comment:
         > this implementation treats aromatic rings as the
         > corresponding Kekule structure with alternating bonds,
         > for purposes of counting "connections".
       Upon further thought, this is the WRONG thing to do.  It
        results in the possibility of a molecule giving two different
        CT values depending on the kekulization.  For example, in the
        old implementation, these two SMILES:
           CC2=CN=C1C3=C(C(C)=C(C=N3)C)C=CC1=C2C
           CC3=CN=C2C1=NC=C(C)C(C)=C1C=CC2=C3C
        which correspond to differentk kekule forms, yield different
        values.
       The new implementation uses consistent (aromatic) bond orders
        for aromatic bonds.

       THIS MEANS THAT THIS IMPLEMENTATION IS NOT BACKWARDS COMPATIBLE.

       Any molecule containing aromatic rings will yield different
       values with this implementation.  The new behavior is the correct
       one, so we're going to live with the breakage.

     **NOTE** this barfs if the molecule contains a second (or
       nth) fragment that is one atom.

  