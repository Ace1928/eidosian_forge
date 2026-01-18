from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class phpass_test(HandlerCase):
    handler = hash.phpass
    known_correct_hashes = [('test12345', '$P$9IQRaTwmfeRo7ud9Fh4E2PdI0S3r.L0'), ('test1', '$H$9aaaaaSXBjgypwqm.JsMssPLiS8YQ00'), ('123456', '$H$9PE8jEklgZhgLmZl5.HYJAzfGCQtzi1'), ('123456', '$H$9pdx7dbOW3Nnt32sikrjAxYFjX8XoK1'), ('thisisalongertestPW', '$P$912345678LIjjb6PhecupozNBmDndU0'), ('JohnRipper', '$P$612345678si5M0DDyPpmRCmcltU/YW/'), ('JohnRipper', '$H$712345678WhEyvy1YWzT4647jzeOmo0'), ('JohnRipper', '$P$B12345678L6Lpt4BxNotVIMILOa9u81'), ('', '$P$7JaFQsPzJSuenezefD/3jHgt5hVfNH0'), ('compL3X!', '$P$FiS0N5L672xzQx1rt1vgdJQRYKnQM9/'), (UPASS_TABLE, '$P$7SMy8VxnfsIy2Sxm7fJxDSdil.h7TW.')]
    known_malformed_hashes = ['$P$9IQRaTwmfeRo7ud9Fh4E2PdI0S3r!L0']