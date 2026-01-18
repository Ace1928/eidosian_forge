import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _add_marker_to_scene(self, marker):
    if self.scn.ngeom >= self.scn.maxgeom:
        raise RuntimeError('Ran out of geoms. maxgeom: %d' % self.scn.maxgeom)
    g = self.scn.geoms[self.scn.ngeom]
    g.dataid = -1
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    g.texid = -1
    g.texuniform = 0
    g.texrepeat[0] = 1
    g.texrepeat[1] = 1
    g.emission = 0
    g.specular = 0.5
    g.shininess = 0.5
    g.reflectance = 0
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size[:] = np.ones(3) * 0.1
    g.mat[:] = np.eye(3)
    g.rgba[:] = np.ones(4)
    for key, value in marker.items():
        if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
            setattr(g, key, value)
        elif isinstance(value, (tuple, list, np.ndarray)):
            attr = getattr(g, key)
            attr[:] = np.asarray(value).reshape(attr.shape)
        elif isinstance(value, str):
            assert key == 'label', 'Only label is a string in mjtGeom.'
            if value is None:
                g.label[0] = 0
            else:
                g.label = value
        elif hasattr(g, key):
            raise ValueError('mjtGeom has attr {} but type {} is invalid'.format(key, type(value)))
        else:
            raise ValueError("mjtGeom doesn't have field %s" % key)
    self.scn.ngeom += 1