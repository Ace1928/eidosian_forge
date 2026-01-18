import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
class cta_pbkdf2_sha1_test(HandlerCase):
    handler = hash.cta_pbkdf2_sha1
    known_correct_hashes = [(u('hashy the ☃'), '$p5k2$1000$ZxK4ZBJCfQg=$jJZVscWtO--p1-xIZl6jhO2LKR0='), ('password', '$p5k2$1$$h1TDLGSw9ST8UMAPeIE13i0t12c='), (UPASS_WAV, '$p5k2$4321$OTg3NjU0MzIx$jINJrSvZ3LXeIbUdrJkRpN62_WQ=')]