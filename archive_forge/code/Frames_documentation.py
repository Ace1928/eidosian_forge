import os
import sys
from Chem import AllChem as Chem
return a ganeric molecule defining the reduced scaffold of the input mol.
    mode can be 'Scaff' or 'RedScaff':

    Scaff	->	chop off the side chains and return the scaffold

    RedScaff	->	remove all linking chains and connect the rings
    directly at the atoms where the linker was
    