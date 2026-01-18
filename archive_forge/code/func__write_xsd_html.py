import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase import Atoms
from ase.utils import writer
def _write_xsd_html(images, connectivity=None):
    ATR, XSD = SetBasicChilds()
    natoms = len(images[0])
    atom_element = images[0].get_chemical_symbols()
    atom_cell = images[0].get_cell()
    atom_positions = images[0].get_positions()
    bonds = list()
    if connectivity is not None:
        for i in range(connectivity.shape[0]):
            for j in range(i + 1, connectivity.shape[0]):
                if connectivity[i, j]:
                    bonds.append([i, j])
    nbonds = len(bonds)
    if not images[0].pbc.all():
        Molecule = SetChild(ATR, 'Molecule', dict(ID='2', NumChildren=str(natoms + nbonds), Name='Lattice=&quot1.0'))
        for x in range(natoms):
            Props = dict(ID=str(x + 3), Name=atom_element[x] + str(x + 1), UserID=str(x + 1), DisplayStyle=CPK_or_BnS(atom_element[x]), XYZ=','.join(('%1.16f' % xi for xi in atom_positions[x])), Components=atom_element[x])
            bondstr = []
            for i, bond in enumerate(bonds):
                if x in bond:
                    bondstr.append(str(i + 3 + natoms))
            if bondstr:
                Props['Connections'] = ','.join(bondstr)
            SetChild(Molecule, 'Atom3d', Props)
        for x in range(nbonds):
            SetChild(Molecule, 'Bond', dict(ID=str(x + 3 + natoms), Connects='%i,%i' % (bonds[x][0] + 3, bonds[x][1] + 3)))
    else:
        atom_positions = np.dot(atom_positions, np.linalg.inv(atom_cell))
        Props = dict(ID='2', Mapping='3', Children=','.join(map(str, range(4, natoms + nbonds + 5))), Normalized='1', Name='SymmSys', UserID=str(natoms + 18), XYZ='0.00000000000000,0.00000000000000,0.000000000000000', OverspecificationTolerance='0.05', PeriodicDisplayType='Original')
        SymmSys = SetChild(ATR, 'SymmetrySystem', Props)
        Props = dict(ID=str(natoms + nbonds + 5), SymmetryDefinition=str(natoms + 4), ActiveSystem='2', NumFamilies='1', OwnsTotalConstraintMapping='1', TotalConstraintMapping='3')
        MappngSet = SetChild(SymmSys, 'MappingSet', Props)
        Props = dict(ID=str(natoms + nbonds + 6), NumImageMappings='0')
        MappngFamily = SetChild(MappngSet, 'MappingFamily', Props)
        Props = dict(ID=str(natoms + len(bonds) + 7), Element='1,0,0,0,0,1,0,0,0,0,1,0', Constraint='1,0,0,0,0,1,0,0,0,0,1,0', MappedObjects=','.join(map(str, range(4, natoms + len(bonds) + 4))), DefectObjects='%i,%i' % (natoms + nbonds + 4, natoms + nbonds + 8), NumImages=str(natoms + len(bonds)), NumDefects='2')
        IdentMappng = SetChild(MappngFamily, 'IdentityMapping', Props)
        SetChild(MappngFamily, 'MappingRepairs', {'NumRepairs': '0'})
        for x in range(natoms):
            Props = dict(ID=str(x + 4), Mapping=str(natoms + len(bonds) + 7), Parent='2', Name=atom_element[x] + str(x + 1), UserID=str(x + 1), DisplayStyle=CPK_or_BnS(atom_element[x]), Components=atom_element[x], XYZ=','.join(['%1.16f' % xi for xi in atom_positions[x]]))
            bondstr = []
            for i, bond in enumerate(bonds):
                if x in bond:
                    bondstr.append(str(i + 4 * natoms + 1))
            if bondstr:
                Props['Connections'] = ','.join(bondstr)
            SetChild(IdentMappng, 'Atom3d', Props)
        for x in range(len(bonds)):
            SetChild(IdentMappng, 'Bond', dict(ID=str(x + 4 + natoms + 1), Mapping=str(natoms + len(bonds) + 7), Parent='2', Connects='%i,%i' % (bonds[x][0] + 4, bonds[x][1] + 4)))
        Props = dict(ID=str(natoms + 4), Parent='2', Children=str(natoms + len(bonds) + 8), DisplayStyle='Solid', XYZ='0.00,0.00,0.00', Color='0,0,0,0', AVector=','.join(['%1.16f' % atom_cell[0, x] for x in range(3)]), BVector=','.join(['%1.16f' % atom_cell[1, x] for x in range(3)]), CVector=','.join(['%1.16f' % atom_cell[2, x] for x in range(3)]), OrientationBase='C along Z, B in YZ plane', Centering='3D Primitive-Centered', Lattice='3D Triclinic', GroupName='GroupName', Operators='1,0,0,0,0,1,0,0,0,0,1,0', DisplayRange='0,1,0,1,0,1', LineThickness='2', CylinderRadius='0.2', LabelAxes='1', ActiveSystem='2', ITNumber='1', LongName='P 1', Qualifier='Origin-1', SchoenfliesName='C1-1', System='Triclinic', Class='1')
        SetChild(IdentMappng, 'SpaceGroup', Props)
        SetChild(IdentMappng, 'ReciprocalLattice3D', dict(ID=str(natoms + len(bonds) + 8), Parent=str(natoms + 4)))
        SetChild(MappngSet, 'InfiniteMapping', dict(ID='3', Element='1,0,0,0,0,1,0,0,0,0,1,0', MappedObjects='2'))
    return (XSD, ATR)