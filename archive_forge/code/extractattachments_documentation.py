import os
from base64 import b64decode
from traitlets import Bool, Unicode
from .base import Preprocessor

        Extract attachments to individual files and
        change references to them.
        E.g.
        '![image.png](attachment:021fdd80.png)'
        becomes
        '![image.png]({path_name}/021fdd80.png)'
        Assumes self.path_name and self.resources_item_key is set properly (usually in preprocess).
        