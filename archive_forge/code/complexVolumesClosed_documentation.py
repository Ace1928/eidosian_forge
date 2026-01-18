from snappy.verify.complex_volume.adjust_torsion import (
from snappy.verify.complex_volume.closed import zero_lifted_holonomy
from snappy.dev.extended_ptolemy import extended
from snappy.dev.extended_ptolemy import giac_rur
import snappy.snap.t3mlite as t3m
from sage.all import (RealIntervalField, ComplexIntervalField,
import sage.all
import re

    Compute all complex volumes from the extended Ptolemy variety for the
    closed manifold M (given as Dehn-filling on 1-cusped manifold).
    Note: not every volume might correspond to a representation factoring
    through the closed manifold. In particular, we get the complex volume
    of the geometric representation of the cusped manifold.
    