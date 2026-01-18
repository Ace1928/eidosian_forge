from Bio.File import as_handle
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.internal_coords import IC_Residue, IC_Chain
from Bio.PDB.vectors import homog_scale_mtx
import numpy as np  # type: ignore
Write hedron assembly to file as OpenSCAD matrices.

    This routine calls both :meth:`.IC_Chain.internal_to_atom_coordinates` and
    :meth:`.IC_Chain.atom_to_internal_coordinates` due to requirements for
    scaling, explicit bonds around rings, and setting the coordinate space of
    the output model.

    Output data format is primarily:

    - matrix for each hedron:
        len1, angle2, len3, atom covalent bond class, flags to indicate
        atom/bond represented in previous hedron (OpenSCAD very slow with
        redundant overlapping elements), flags for bond features
    - transform matrices to assemble each hedron into residue dihedra sets
    - transform matrices for each residue to position in chain

    OpenSCAD software is included in this Python file to process these
    matrices into a model suitable for a 3D printing project.

    :param entity: Biopython PDB :class:`.Structure` entity
        structure data to export
    :param file: Bipoython :func:`.as_handle` filename or open file pointer
        file to write data to
    :param float scale:
        units (usually mm) per angstrom for STL output, written in output
    :param str pdbid:
        PDB idcode, written in output. Defaults to '0PDB' if not supplied
        and no 'idcode' set in entity
    :param bool backboneOnly: default False.
        Do not output side chain data past Cbeta if True
    :param bool includeCode: default True.
        Include OpenSCAD software (inline below) so output file can be loaded
        into OpenSCAD; if False, output data matrices only
    :param float maxPeptideBond: Optional default None.
        Override the cut-off in IC_Chain class (default 1.4) for detecting
        chain breaks.  If your target has chain breaks, pass a large number
        here to create a very long 'bond' spanning the break.
    :param int start,fin: default None
        Parameters for internal_to_atom_coords() to limit chain segment.
    :param str handle: default 'protein'
        name for top level of generated OpenSCAD matrix structure

    See :meth:`.IC_Residue.set_flexible` to set flags for specific residues to
    have rotatable bonds, and :meth:`.IC_Residue.set_hbond` to include cavities
    for small magnets to work as hydrogen bonds.
    See <https://www.thingiverse.com/thing:3957471> for implementation example.

    The OpenSCAD code explicitly creates spheres and cylinders to
    represent atoms and bonds in a 3D model.  Options are available
    to support rotatable bonds and magnetic hydrogen bonds.

    Matrices are written to link, enumerate and describe residues,
    dihedra, hedra, and chains, mirroring contents of the relevant IC_*
    data structures.

    The OpenSCAD matrix of hedra has additional information as follows:

    * the atom and bond state (single, double, resonance) are logged
        so that covalent radii may be used for atom spheres in the 3D models

    * bonds and atoms are tracked so that each is only created once

    * bond options for rotation and magnet holders for hydrogen bonds
        may be specified (see :meth:`.IC_Residue.set_flexible` and
        :meth:`.IC_Residue.set_hbond` )

    Note the application of :data:`Bio.PDB.internal_coords.IC_Chain.MaxPeptideBond`
    :  missing residues may be linked (joining chain segments with arbitrarily
    long bonds) by setting this to a large value.

    Note this uses the serial assembly per residue, placing each residue at
    the origin and supplying the coordinate space transform to OpenaSCAD

    All ALTLOC (disordered) residues and atoms are written to the output
    model.  (see :data:`Bio.PDB.internal_coords.IC_Residue.no_altloc`)
    