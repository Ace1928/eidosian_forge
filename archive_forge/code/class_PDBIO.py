import warnings
from Bio import BiopythonWarning
from Bio.PDB.PDBExceptions import PDBIOException
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.Data.IUPACData import atom_weights
class PDBIO(StructureIO):
    """Write a Structure object (or a subset of a Structure object) as a PDB or PQR file.

    Examples
    --------
    >>> from Bio.PDB import PDBParser
    >>> from Bio.PDB.PDBIO import PDBIO
    >>> parser = PDBParser()
    >>> structure = parser.get_structure("1a8o", "PDB/1A8O.pdb")
    >>> io=PDBIO()
    >>> io.set_structure(structure)
    >>> io.save("bio-pdb-pdbio-out.pdb")
    >>> import os
    >>> os.remove("bio-pdb-pdbio-out.pdb")  # tidy up


    """

    def __init__(self, use_model_flag=0, is_pqr=False):
        """Create the PDBIO object.

        :param use_model_flag: if 1, force use of the MODEL record in output.
        :type use_model_flag: int
        :param is_pqr: if True, build PQR file. Otherwise build PDB file.
        :type is_pqr: Boolean
        """
        self.use_model_flag = use_model_flag
        self.is_pqr = is_pqr

    def _get_atom_line(self, atom, hetfield, segid, atom_number, resname, resseq, icode, chain_id, charge='  '):
        """Return an ATOM PDB string (PRIVATE)."""
        if hetfield != ' ':
            record_type = 'HETATM'
        else:
            record_type = 'ATOM  '
        try:
            atom_number = int(atom_number)
        except ValueError:
            raise ValueError(f'{atom_number!r} is not a number.Atom serial numbers must be numerical If you are converting from an mmCIF structure, try using preserve_atom_numbering=False')
        if atom_number > 99999:
            raise ValueError(f"Atom serial number ('{atom_number}') exceeds PDB format limit.")
        if atom.element:
            element = atom.element.strip().upper()
            if element.capitalize() not in atom_weights and element != 'X':
                raise ValueError(f'Unrecognised element {atom.element}')
            element = element.rjust(2)
        else:
            element = '  '
        name = atom.fullname.strip()
        if len(name) < 4 and name[:1].isalpha() and (len(element.strip()) < 2):
            name = ' ' + name
        altloc = atom.altloc
        x, y, z = atom.coord
        if not self.is_pqr:
            bfactor = atom.bfactor
            try:
                occupancy = f'{atom.occupancy:6.2f}'
            except (TypeError, ValueError):
                if atom.occupancy is None:
                    occupancy = ' ' * 6
                    warnings.warn(f'Missing occupancy in atom {atom.full_id!r} written as blank', BiopythonWarning)
                else:
                    raise ValueError(f'Invalid occupancy value: {atom.occupancy!r}') from None
            args = (record_type, atom_number, name, altloc, resname, chain_id, resseq, icode, x, y, z, occupancy, bfactor, segid, element, charge)
            return _ATOM_FORMAT_STRING % args
        else:
            try:
                pqr_charge = f'{atom.pqr_charge:7.4f}'
            except (TypeError, ValueError):
                if atom.pqr_charge is None:
                    pqr_charge = ' ' * 7
                    warnings.warn(f'Missing PQR charge in atom {atom.full_id} written as blank', BiopythonWarning)
                else:
                    raise ValueError(f'Invalid PQR charge value: {atom.pqr_charge!r}') from None
            try:
                radius = f'{atom.radius:6.4f}'
            except (TypeError, ValueError):
                if atom.radius is None:
                    radius = ' ' * 6
                    warnings.warn(f'Missing radius in atom {atom.full_id} written as blank', BiopythonWarning)
                else:
                    raise ValueError(f'Invalid radius value: {atom.radius}') from None
            args = (record_type, atom_number, name, altloc, resname, chain_id, resseq, icode, x, y, z, pqr_charge, radius, element)
            return _PQR_ATOM_FORMAT_STRING % args

    def save(self, file, select=_select, write_end=True, preserve_atom_numbering=False):
        """Save structure to a file.

        :param file: output file
        :type file: string or filehandle

        :param select: selects which entities will be written.
        :type select: object

        Typically select is a subclass of L{Select}, it should
        have the following methods:

         - accept_model(model)
         - accept_chain(chain)
         - accept_residue(residue)
         - accept_atom(atom)

        These methods should return 1 if the entity is to be
        written out, 0 otherwise.

        Typically select is a subclass of L{Select}.
        """
        if isinstance(file, str):
            fhandle = open(file, 'w')
        else:
            fhandle = file
        get_atom_line = self._get_atom_line
        if len(self.structure) > 1 or self.use_model_flag:
            model_flag = 1
        else:
            model_flag = 0
        for model in self.structure.get_list():
            if not select.accept_model(model):
                continue
            model_residues_written = 0
            if not preserve_atom_numbering:
                atom_number = 1
            if model_flag:
                fhandle.write(f'MODEL      {model.serial_num}\n')
            for chain in model.get_list():
                if not select.accept_chain(chain):
                    continue
                chain_id = chain.id
                if len(chain_id) > 1:
                    e = f"Chain id ('{chain_id}') exceeds PDB format limit."
                    raise PDBIOException(e)
                chain_residues_written = 0
                for residue in chain.get_unpacked_list():
                    if not select.accept_residue(residue):
                        continue
                    hetfield, resseq, icode = residue.id
                    resname = residue.resname
                    segid = residue.segid
                    resid = residue.id[1]
                    if resid > 9999:
                        e = f"Residue number ('{resid}') exceeds PDB format limit."
                        raise PDBIOException(e)
                    for atom in residue.get_unpacked_list():
                        if not select.accept_atom(atom):
                            continue
                        chain_residues_written = 1
                        model_residues_written = 1
                        if preserve_atom_numbering:
                            atom_number = atom.serial_number
                        try:
                            s = get_atom_line(atom, hetfield, segid, atom_number, resname, resseq, icode, chain_id)
                        except Exception as err:
                            raise PDBIOException(f'Error when writing atom {atom.full_id}') from err
                        else:
                            fhandle.write(s)
                            atom_number += 1
                if chain_residues_written:
                    fhandle.write(_TER_FORMAT_STRING % (atom_number, resname, chain_id, resseq, icode))
            if model_flag and model_residues_written:
                fhandle.write('ENDMDL\n')
        if write_end:
            fhandle.write('END   \n')
        if isinstance(file, str):
            fhandle.close()