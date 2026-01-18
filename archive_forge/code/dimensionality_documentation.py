import os
import warnings
from ase.io import iread
from ase.geometry.dimensionality import analyze_dimensionality
Analyze the dimensionality of the bonded clusters in a structure, using
    the scoring parameter described in:

    "Definition of a scoring parameter to identify low-dimensional materials
    components",  P.M. Larsen, M. Pandey, M. Strange, and K. W. Jacobsen
    Phys. Rev. Materials 3 034003, 2019,
    https://doi.org/10.1103/PhysRevMaterials.3.034003
    https://arxiv.org/abs/1808.02114

    A score in the range [0-1] is assigned to each possible dimensionality
    classification. The scores sum to 1. A bonded cluster can be a molecular
    (0D), chain (1D), layer (2D), or bulk (3D) cluster. Mixed dimensionalities,
    such as 0D+3D are possible. Input files may use any format supported by
    ASE.

    Example usage:

    * ase dimensionality --display-all structure.cif
    * ase dimensionality structure1.cif structure2.cif

    For each structure the following data is printed:

    * type             - the dimensionalities present
    * score            - the score of the classification
    * a                - the start of the k-interval (see paper)
    * b                - the end of the k-interval (see paper)
    * component counts - the number of clusters with each dimensionality type

    If the `--display-all` option is used, all dimensionality classifications
    are displayed.
    