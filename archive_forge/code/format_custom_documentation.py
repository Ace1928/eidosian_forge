import codecs
from breezy import errors
from breezy.lazy_regex import lazy_compile
from breezy.revision import NULL_REVISION
from breezy.version_info_formats import VersionInfoBuilder, create_date_str
Create a version file based on a custom template.