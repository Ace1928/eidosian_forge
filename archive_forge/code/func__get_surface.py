import os
from random import choice
from itertools import combinations
import snappy
from plink import LinkManager
from .twister_core import build_bundle, build_splitting, twister_version
def _get_surface(surface):
    if isinstance(surface, tuple) and len(surface) == 2 and isinstance(surface[0], int) and isinstance(surface[1], int):
        return LP_surface(surface[0], surface[1])
    if isinstance(surface, basestring):
        if surface.startswith('# A Twister surface file'):
            return surface
        if surface in surface_database:
            surface = os.path.join(surface_database_path, surface)
        try:
            lines = open(surface, 'r').readlines()
        except IOError:
            raise IOError('Unable to open %s' % surface)
        contents = ''.join(lines)
        if contents.startswith('% Virtual Link Projection\n'):
            LM = LinkManager()
            LM._from_string(contents)
            return LM.Twister_surface_file()
        if contents.startswith('# A Twister surface file'):
            return contents
        raise TypeError('Not a Twister surface file.')
    raise TypeError('Surfaces can only be loaded from a string or pair of integers (genus,boundary).')