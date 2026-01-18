import numpy as np
from Bio.PDB.PDBExceptions import PDBException
class QCPSuperimposer:
    """Quaternion Characteristic Polynomial (QCP) Superimposer.

    QCPSuperimposer finds the best rotation and translation to put
    two point sets on top of each other (minimizing the RMSD). This is
    eg. useful to superimposing 3D structures of proteins.

    QCP stands for Quaternion Characteristic Polynomial, which is used
    in the algorithm.

    Reference:

    Douglas L Theobald (2005), "Rapid calculation of RMSDs using a
    quaternion-based characteristic polynomial.", Acta Crystallogr
    A 61(4):478-480
    """

    def __init__(self):
        """Initialize the class."""
        self._reset_properties()

    def _reset_properties(self):
        """Reset all relevant properties to None to avoid conflicts between runs."""
        self.reference_coords = None
        self.coords = None
        self.transformed_coords = None
        self.rot = None
        self.tran = None
        self.rms = None
        self.init_rms = None

    def set_atoms(self, fixed, moving):
        """Prepare alignment between two atom lists.

        Put (translate/rotate) the atoms in fixed on the atoms in
        moving, in such a way that the RMSD is minimized.

        :param fixed: list of (fixed) atoms
        :param moving: list of (moving) atoms
        :type fixed,moving: [L{Atom}, L{Atom},...]
        """
        assert len(fixed) == len(moving), 'Fixed and moving atom lists differ in size'
        fix_coord = np.array([a.get_coord() for a in fixed], dtype=np.float64)
        mov_coord = np.array([a.get_coord() for a in moving], dtype=np.float64)
        self.set(fix_coord, mov_coord)
        self.run()
        self.rms = self.get_rms()
        self.rotran = self.get_rotran()

    def apply(self, atom_list):
        """Apply the QCP rotation matrix/translation vector to a set of atoms."""
        if self.rotran is None:
            raise PDBException('No transformation has been calculated yet')
        rot, tran = self.rotran
        for atom in atom_list:
            atom.transform(rot, tran)

    def set(self, reference_coords, coords):
        """Set the coordinates to be superimposed.

        coords will be put on top of reference_coords.

        - reference_coords: an NxDIM array
        - coords: an NxDIM array

        DIM is the dimension of the points, N is the number
        of points to be superimposed.
        """
        self._reset_properties()
        self.reference_coords = reference_coords
        self.coords = coords
        self._natoms, n_dim = coords.shape
        if reference_coords.shape != coords.shape:
            raise PDBException('Coordinates must have the same dimensions.')
        if n_dim != 3:
            raise PDBException('Coordinates must be Nx3 arrays.')

    def run(self):
        """Superimpose the coordinate sets."""
        if self.coords is None or self.reference_coords is None:
            raise PDBException('No coordinates set.')
        coords = self.coords.copy()
        coords_ref = self.reference_coords.copy()
        com_coords = np.mean(coords, axis=0)
        com_ref = np.mean(coords_ref, axis=0)
        coords -= com_coords
        coords_ref -= com_ref
        self.rms, self.rot, _ = qcp(coords_ref, coords, self._natoms)
        self.tran = com_ref - np.dot(com_coords, self.rot)

    def get_transformed(self):
        """Get the transformed coordinate set."""
        if self.coords is None or self.reference_coords is None:
            raise PDBException('No coordinates set.')
        if self.rot is None:
            raise PDBException('Nothing is superimposed yet.')
        self.transformed_coords = np.dot(self.coords, self.rot) + self.tran
        return self.transformed_coords

    def get_rotran(self):
        """Return right multiplying rotation matrix and translation vector."""
        if self.rot is None:
            raise PDBException('Nothing is superimposed yet.')
        return (self.rot, self.tran)

    def get_init_rms(self):
        """Return the root mean square deviation of untransformed coordinates."""
        if self.coords is None:
            raise PDBException('No coordinates set yet.')
        if self.init_rms is None:
            diff = self.coords - self.reference_coords
            self.init_rms = np.sqrt(np.sum(np.sum(diff * diff, axis=1) / self._natoms))
        return self.init_rms

    def get_rms(self):
        """Root mean square deviation of superimposed coordinates."""
        if self.rms is None:
            raise PDBException('Nothing superimposed yet.')
        return self.rms