from rdkit.Chem.Pharm2D import SigFactory, Utils
from rdkit.RDLogger import logger
 generates a 2D fingerprint for a molecule using the
   parameters in _sig_

   **Arguments**

     - mol: the molecule for which the signature should be generated

     - sigFactory : the SigFactory object with signature parameters
       NOTE: no preprocessing is carried out for _sigFactory_.
             It *must* be pre-initialized.

     - perms: (optional) a sequence of permutation indices limiting which
       pharmacophore combinations are allowed

     - dMat: (optional) the distance matrix to be used

     - bitInfo: (optional) used to return the atoms involved in the bits

  