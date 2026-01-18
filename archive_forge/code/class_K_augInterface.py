import os
from pyomo.environ import SolverFactory
from pyomo.common.tempfiles import TempfileManager
class K_augInterface(object):
    """
    k_aug and dot_sens store information in the user's filesystem,
    some of which is mandatory for subsequent calls.
    This class ensures that calls to these executables happen in
    temporary directories. The resulting files are immediately read
    and cached as attributes of this object, and the temporary
    directories deleted. If we have cached files that can be used
    by a subsequent call to k_aug or dot_sens, we write them just
    before calling the executable, and they are deleted along with
    the temporary directory.

    NOTE: only covers dsdp_mode for now.
    """

    def __init__(self, k_aug=None, dot_sens=None):
        if k_aug is None:
            k_aug = SolverFactory('k_aug')
            k_aug.options['dsdp_mode'] = ''
        if dot_sens is None:
            dot_sens = SolverFactory('dot_sens')
            dot_sens.options['dsdp_mode'] = ''
        if k_aug.available():
            self._k_aug = k_aug
        else:
            raise RuntimeError('k_aug is not available.')
        if dot_sens.available():
            self._dot_sens = dot_sens
        else:
            raise RuntimeError('dot_sens is not available')
        self.data = {fname: None for fname in known_files}

    def k_aug(self, model, **kwargs):
        with InTempDir():
            results = self._k_aug.solve(model, **kwargs)
            for fname in known_files:
                if os.path.exists(fname):
                    with open(fname, 'r') as fp:
                        self.data[fname] = fp.read()
        return results

    def dot_sens(self, model, **kwargs):
        with InTempDir():
            for fname, contents in self.data.items():
                if contents is not None:
                    with open(fname, 'w') as fp:
                        fp.write(contents)
            results = self._dot_sens.solve(model, **kwargs)
            for fname in known_files:
                if os.path.exists(fname):
                    with open(fname, 'r') as fp:
                        self.data[fname] = fp.read()
        return results

    def set_k_aug_options(self, **options):
        for key, val in options.items():
            self._k_aug.options[key] = val

    def set_dot_sens_options(self, **options):
        for key, val in options.items():
            self._dot_sens.options[key] = val