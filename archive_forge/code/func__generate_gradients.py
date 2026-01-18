from multiprocessing import Pool, cpu_count
import os.path as op
import numpy as np
import nibabel as nb
from ... import logging
from ..base import (
from .base import DipyBaseInterface
def _generate_gradients(ndirs=64, values=[1000, 3000], nb0s=1):
    """
    Automatically generate a `gradient table
    <http://nipy.org/dipy/examples_built/gradients_spheres.html#example-gradients-spheres>`_

    """
    import numpy as np
    from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
    from dipy.core.gradients import gradient_table
    theta = np.pi * np.random.rand(ndirs)
    phi = 2 * np.pi * np.random.rand(ndirs)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    values = np.atleast_1d(values).tolist()
    vertices = hsph_updated.vertices
    bvecs = vertices.copy()
    bvals = np.ones(vertices.shape[0]) * values[0]
    for v in values[1:]:
        bvecs = np.vstack((bvecs, vertices))
        bvals = np.hstack((bvals, v * np.ones(vertices.shape[0])))
    for i in range(0, nb0s):
        bvals = bvals.tolist()
        bvals.insert(0, 0)
        bvecs = bvecs.tolist()
        bvecs.insert(0, np.zeros(3))
    return gradient_table(bvals, bvecs)