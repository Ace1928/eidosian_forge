import re
import os
from io import StringIO
import subprocess
import warnings
from Bio.PDB.AbstractPropertyMap import AbstractResiduePropertyMap
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1, residue_sasa_scales
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
class DSSP(AbstractResiduePropertyMap):
    """Run DSSP and parse secondary structure and accessibility.

    Run DSSP on a PDB/mmCIF file, and provide a handle to the
    DSSP secondary structure and accessibility.

    **Note** that DSSP can only handle one model.

    Examples
    --------
    How DSSP could be used::

        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
        p = PDBParser()
        structure = p.get_structure("1MOT", "/local-pdb/1mot.pdb")
        model = structure[0]
        dssp = DSSP(model, "/local-pdb/1mot.pdb")
        # DSSP data is accessed by a tuple (chain_id, res_id)
        a_key = list(dssp.keys())[2]
        # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
        # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
        # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
        dssp[a_key]

    """

    def __init__(self, model, in_file, dssp='dssp', acc_array='Sander', file_type=''):
        """Create a DSSP object.

        Parameters
        ----------
        model : Model
            The first model of the structure
        in_file : string
            Either a PDB file or a DSSP file.
        dssp : string
            The dssp executable (ie. the argument to subprocess)
        acc_array : string
            Accessible surface area (ASA) from either Miller et al. (1987),
            Sander & Rost (1994), Wilke: Tien et al. 2013, or Ahmad et al.
            (2003) as string Sander/Wilke/Miller/Ahmad. Defaults to Sander.
        file_type: string
            File type switch: either PDB, MMCIF or DSSP. Inferred from the
            file extension by default.

        """
        self.residue_max_acc = residue_max_acc[acc_array]
        if file_type == '':
            file_type = os.path.splitext(in_file)[1][1:]
        file_type = file_type.upper()
        if file_type == 'CIF':
            file_type = 'MMCIF'
        assert file_type in ['PDB', 'MMCIF', 'DSSP'], 'File type must be PDB, mmCIF or DSSP'
        if file_type == 'PDB' or file_type == 'MMCIF':
            try:
                version_string = subprocess.check_output([dssp, '--version'], universal_newlines=True)
                dssp_version = re.search('\\s*([\\d.]+)', version_string).group(1)
                dssp_dict, dssp_keys = dssp_dict_from_pdb_file(in_file, dssp, dssp_version)
            except FileNotFoundError:
                if dssp == 'dssp':
                    dssp = 'mkdssp'
                elif dssp == 'mkdssp':
                    dssp = 'dssp'
                else:
                    raise
                version_string = subprocess.check_output([dssp, '--version'], universal_newlines=True)
                dssp_version = re.search('\\s*([\\d.]+)', version_string).group(1)
                dssp_dict, dssp_keys = dssp_dict_from_pdb_file(in_file, dssp, dssp_version)
        elif file_type == 'DSSP':
            dssp_dict, dssp_keys = make_dssp_dict(in_file)
        dssp_map = {}
        dssp_list = []

        def resid2code(res_id):
            """Serialize a residue's resseq and icode for easy comparison."""
            return f'{res_id[1]}{res_id[2]}'
        if file_type == 'MMCIF' and version(dssp_version) < version('4.0.0'):
            mmcif_dict = MMCIF2Dict(in_file)
            mmcif_chain_dict = {}
            for i, c in enumerate(mmcif_dict['_atom_site.label_asym_id']):
                if c not in mmcif_chain_dict:
                    mmcif_chain_dict[c] = mmcif_dict['_atom_site.auth_asym_id'][i]
            dssp_mapped_keys = []
        for key in dssp_keys:
            chain_id, res_id = key
            if file_type == 'MMCIF' and version(dssp_version) < version('4.0.0'):
                chain_id = mmcif_chain_dict[chain_id]
                dssp_mapped_keys.append((chain_id, res_id))
            chain = model[chain_id]
            try:
                res = chain[res_id]
            except KeyError:
                res_seq_icode = resid2code(res_id)
                for r in chain:
                    if r.id[0] not in (' ', 'W'):
                        if resid2code(r.id) == res_seq_icode:
                            res = r
                            break
                else:
                    raise KeyError(res_id) from None
            if res.is_disordered() == 2:
                for rk in res.disordered_get_id_list():
                    altloc = res.child_dict[rk].get_list()[0].get_altloc()
                    if altloc in tuple('A1 '):
                        res.disordered_select(rk)
                        break
                else:
                    res.disordered_select(res.disordered_get_id_list()[0])
            elif res.is_disordered() == 1:
                altlocs = {a.get_altloc() for a in res.get_unpacked_list()}
                if altlocs.isdisjoint('A1 '):
                    res_seq_icode = resid2code(res_id)
                    for r in chain:
                        if r.id[0] not in (' ', 'W'):
                            if resid2code(r.id) == res_seq_icode and r.get_list()[0].get_altloc() in tuple('A1 '):
                                res = r
                                break
            aa, ss, acc, phi, psi, dssp_index, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy = dssp_dict[key]
            res.xtra['SS_DSSP'] = ss
            res.xtra['EXP_DSSP_ASA'] = acc
            res.xtra['PHI_DSSP'] = phi
            res.xtra['PSI_DSSP'] = psi
            res.xtra['DSSP_INDEX'] = dssp_index
            res.xtra['NH_O_1_RELIDX_DSSP'] = NH_O_1_relidx
            res.xtra['NH_O_1_ENERGY_DSSP'] = NH_O_1_energy
            res.xtra['O_NH_1_RELIDX_DSSP'] = O_NH_1_relidx
            res.xtra['O_NH_1_ENERGY_DSSP'] = O_NH_1_energy
            res.xtra['NH_O_2_RELIDX_DSSP'] = NH_O_2_relidx
            res.xtra['NH_O_2_ENERGY_DSSP'] = NH_O_2_energy
            res.xtra['O_NH_2_RELIDX_DSSP'] = O_NH_2_relidx
            res.xtra['O_NH_2_ENERGY_DSSP'] = O_NH_2_energy
            resname = res.get_resname()
            try:
                rel_acc = acc / self.residue_max_acc[resname]
            except KeyError:
                rel_acc = 'NA'
            else:
                if rel_acc > 1.0:
                    rel_acc = 1.0
            res.xtra['EXP_DSSP_RASA'] = rel_acc
            resname = protein_letters_3to1.get(resname, 'X')
            if resname == 'C':
                if _dssp_cys.match(aa):
                    aa = 'C'
            if resname != aa and (res.id[0] == ' ' or aa != 'X'):
                raise PDBException(f'Structure/DSSP mismatch at {res}')
            dssp_vals = (dssp_index, aa, ss, rel_acc, phi, psi, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
            dssp_map[chain_id, res_id] = dssp_vals
            dssp_list.append(dssp_vals)
        if file_type == 'MMCIF' and version(dssp_version) < version('4.0.0'):
            dssp_keys = dssp_mapped_keys
        AbstractResiduePropertyMap.__init__(self, dssp_map, dssp_keys, dssp_list)