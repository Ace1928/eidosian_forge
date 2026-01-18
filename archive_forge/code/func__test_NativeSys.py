from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_NativeSys(NativeSys, **kwargs):
    native = NativeSys.from_callback(vdp_f, 2, 1)
    assert native.ny == 2
    assert len(native.params) == 1
    xout, yout, info = native.integrate([0, 1, 2], [1, 0], params=[2.0], **kwargs)
    ref = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(yout, ref)
    if 'nfev' in info:
        assert info['nfev'] > 0