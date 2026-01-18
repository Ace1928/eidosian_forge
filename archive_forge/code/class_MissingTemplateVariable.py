import codecs
from breezy import errors
from breezy.lazy_regex import lazy_compile
from breezy.revision import NULL_REVISION
from breezy.version_info_formats import VersionInfoBuilder, create_date_str
class MissingTemplateVariable(errors.BzrError):
    _fmt = 'Variable {%(name)s} is not available.'

    def __init__(self, name):
        self.name = name