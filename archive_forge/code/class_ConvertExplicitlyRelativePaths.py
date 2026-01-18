import os
from pandocfilters import Image, applyJSONFilters  # type:ignore[import-untyped]
from nbconvert.utils.base import NbConvertBase
from nbconvert.utils.pandoc import pandoc
class ConvertExplicitlyRelativePaths(NbConvertBase):
    """A converter that handles relative path references."""

    def __init__(self, texinputs=None, **kwargs):
        """Initialize the converter."""
        self.nb_dir = os.path.abspath(texinputs) if texinputs else ''
        self.ancestor_dirs = self.nb_dir.split('/')
        super().__init__(**kwargs)

    def __call__(self, source):
        """Invoke the converter."""
        if self.nb_dir:
            return applyJSONFilters([self.action], source)
        return source

    def action(self, key, value, frmt, meta):
        """Perform the action."""
        if key == 'Image':
            attr, caption, [filename, typedef] = value
            if filename[:2] == './':
                filename = filename[2:]
            elif filename[:3] == '../':
                n_up = 0
                while filename[:3] == '../':
                    n_up += 1
                    filename = filename[3:]
                ancestors = '/'.join(self.ancestor_dirs[:-n_up]) + '/'
                filename = ancestors + filename
            return Image(attr, caption, [filename, typedef])
        return None