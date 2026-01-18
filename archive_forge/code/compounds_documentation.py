from ase.lattice.cubic import DiamondFactory, SimpleCubicFactory
from ase.lattice.tetragonal import SimpleTetragonalFactory
from ase.lattice.triclinic import TriclinicFactory
from ase.lattice.hexagonal import HexagonalFactory
A factory for creating hematite (Fe2O3) lattices.
     With hexagonal unit cell.
     Blake R L, Hessevick R E, Zoltai T, Finger L W
     American Mineralogist 51 (1966) 123-129
     5.038 5.038 13.772 90 90 120 R-3c
     Fe       0 0 .3553  .0080  .0080 .00029  .0040      0      0
     O    .3059 0   1/4  .0068  .0083 .00046  .0042 .00058  .0012

     Example:
     #!/usr/bin/env python3
     from ase.lattice.hexagonal import *
     from ase.lattice.compounds import *
     import ase.io as io
     from ase import Atoms, Atom

     index1=1
     index2=1
     index3=1
     mya = 5.038
     myb = 5.038
     myc = 13.772
     myalpha = 90
     mybeta = 90
     mygamma = 120
     gra = HEX_Fe2O3(symbol = ('Fe', 'O'),
     latticeconstant={'a':mya,'b':myb, 'c':myc,
     'alpha':myalpha,
     'beta':mybeta,
     'gamma':mygamma},
     size=(index1,index2,index3))
     io.write('hexaFe2O3.xyz', gra, format='xyz')

     