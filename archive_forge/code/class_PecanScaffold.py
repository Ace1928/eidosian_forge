import sys
import os
import re
import pkg_resources
from string import Template
class PecanScaffold(object):
    """
    A base Pecan scaffold.  New scaffolded implementations should extend this
    class and define a ``_scaffold_dir`` attribute, e.g.,

    class CoolAddOnScaffold(PecanScaffold):

        _scaffold_dir = ('package', os.path.join('scaffolds', 'scaffold_name'))

    ...where...

        pkg_resources.resource_listdir(_scaffold_dir[0], _scaffold_dir[1]))

    ...points to some scaffold directory root.
    """

    def normalize_output_dir(self, dest):
        return os.path.abspath(os.path.normpath(dest))

    def normalize_pkg_name(self, dest):
        return _bad_chars_re.sub('', dest.lower())

    def copy_to(self, dest, **kw):
        output_dir = self.normalize_output_dir(dest)
        pkg_name = self.normalize_pkg_name(dest)
        copy_dir(self._scaffold_dir, output_dir, {'package': pkg_name}, **kw)