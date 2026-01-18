from ase.units import Bohr
class TurbomoleFormatError(ValueError):
    default_message = 'Data format in file does not correspond to known Turbomole gradient format'

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            ValueError.__init__(self, *args, **kwargs)
        else:
            ValueError.__init__(self, self.default_message)