import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
class argon2_argon2_cffi_test(_base_argon2_test.create_backend_case('argon2_cffi')):
    known_correct_hashes = _base_argon2_test.known_correct_hashes + [('password', '$argon2i$m=65536,t=2,p=4$c29tZXNhbHQAAAAAAAAAAA$QWLzI4TY9HkL2ZTLc8g6SinwdhZewYrzz9zxCo0bkGY'), ('password', '$argon2i$v=19$m=65536,t=2,p=4$c29tZXNhbHQ$IMit9qkFULCMA/ViizL57cnTLOa5DiVM9eMwpAvPwr4'), ('password', '$argon2d$v=19$m=65536,t=2,p=4$c29tZXNhbHQ$cZn5d+rFh+ZfuRhm2iGUGgcrW5YLeM6q7L3vBsdmFA0'), ('password', '$argon2id$v=19$m=65536,t=2,p=4$c29tZXNhbHQ$GpZ3sK/oH9p7VIiV56G/64Zo/8GaUw434IimaPqxwCo'), ('password\x00', '$argon2i$v=19$m=65536,t=2,p=4$c29tZXNhbHQ$Vpzuc0v0SrP88LcVvmg+z5RoOYpMDKH/lt6O+CZabIQ')]
    known_correct_hashes.extend(((info['secret'], info['hash']) for info in reference_data if info['logM'] <= (18 if TEST_MODE('full') else 16)))