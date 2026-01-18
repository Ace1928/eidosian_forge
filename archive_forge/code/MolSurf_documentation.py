import bisect
import numpy
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors, rdPartialCharges
 DEPRECATED: this has been reimplmented in C++
   calculates the polar surface area of a molecule based upon fragments

   Algorithm in:
    P. Ertl, B. Rohde, P. Selzer
     Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-based
     Contributions and Its Application to the Prediction of Drug Transport
     Properties, J.Med.Chem. 43, 3714-3717, 2000

   Implementation based on the Daylight contrib program tpsa.c
  