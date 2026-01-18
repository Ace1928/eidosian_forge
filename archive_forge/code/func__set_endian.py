import numpy
def _set_endian(self, c):
    """Set endian to big (c='>') or little (c='<') or native (c='=')

        :Parameters:
          `c` : string
            The endian-ness to use when reading from this file.
        """
    if c in '<>@=':
        if c == '@':
            c = '='
        self._endian = c
    else:
        raise ValueError('Cannot set endian-ness')