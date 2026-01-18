from collections.abc import Sequence
from traits import __version__ as traits_version
import traits.api as traits
from traits.api import TraitType, Unicode
from traits.trait_base import _Undefined
from pathlib import Path
from ...utils.filemanip import path_resolve
class BasePath(TraitType):
    """Defines a trait whose value must be a valid filesystem path."""
    exists = False
    resolve = False
    _is_file = False
    _is_dir = False

    @property
    def info_text(self):
        """Create the trait's general description."""
        info_text = 'a pathlike object or string'
        if any((self.exists, self._is_file, self._is_dir)):
            info_text += ' representing a'
            if self.exists:
                info_text += 'n existing'
            if self._is_file:
                info_text += ' file'
            elif self._is_dir:
                info_text += ' directory'
            else:
                info_text += ' file or directory'
        return info_text

    def __init__(self, value=Undefined, exists=False, resolve=False, **metadata):
        """Create a BasePath trait."""
        self.exists = exists
        self.resolve = resolve
        super(BasePath, self).__init__(value, **metadata)

    def validate(self, objekt, name, value, return_pathlike=False):
        """Validate a value change."""
        try:
            value = Path(value)
        except Exception:
            self.error(objekt, name, str(value))
        if self.exists:
            if not value.exists():
                self.error(objekt, name, str(value))
            if self._is_file and (not value.is_file()):
                self.error(objekt, name, str(value))
            if self._is_dir and (not value.is_dir()):
                self.error(objekt, name, str(value))
        if self.resolve:
            value = path_resolve(value, strict=self.exists)
        if not return_pathlike:
            value = str(value)
        return value