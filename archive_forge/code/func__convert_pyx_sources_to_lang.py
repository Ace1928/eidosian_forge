import re
import functools
import distutils.core
import distutils.errors
import distutils.extension
from .monkey import get_unpatched
def _convert_pyx_sources_to_lang(self):
    """
        Replace sources with .pyx extensions to sources with the target
        language extension. This mechanism allows language authors to supply
        pre-converted sources but to prefer the .pyx sources.
        """
    if _have_cython():
        return
    lang = self.language or ''
    target_ext = '.cpp' if lang.lower() == 'c++' else '.c'
    sub = functools.partial(re.sub, '.pyx$', target_ext)
    self.sources = list(map(sub, self.sources))