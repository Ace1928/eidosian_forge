import numpy as np
from ase.atoms import Atoms
class Quaternions(Atoms):

    def __init__(self, *args, **kwargs):
        quaternions = None
        if 'quaternions' in kwargs:
            quaternions = np.array(kwargs['quaternions'])
            del kwargs['quaternions']
        Atoms.__init__(self, *args, **kwargs)
        if quaternions is not None:
            self.set_array('quaternions', quaternions, shape=(4,))
            self.set_shapes(np.array([[3, 2, 1]] * len(self)))

    def set_shapes(self, shapes):
        self.set_array('shapes', shapes, shape=(3,))

    def set_quaternions(self, quaternions):
        self.set_array('quaternions', quaternions, quaternion=(4,))

    def get_shapes(self):
        return self.get_array('shapes')

    def get_quaternions(self):
        return self.get_array('quaternions').copy()