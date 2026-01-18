import pytest  # NOQA
from .roundtrip import round_trip
class TestProgramConfig:

    def test_application_arguments(self):
        round_trip('\n        args:\n          username: anthon\n          passwd: secret\n          fullname: Anthon van der Neut\n          tmux:\n            session-name: test\n          loop:\n            wait: 10\n        ')

    def test_single(self):
        round_trip("\n        # default arguments for the program\n        args:  # needed to prevent comment wrapping\n        # this should be your username\n          username: anthon\n          passwd: secret        # this is plaintext don't reuse # important/system passwords\n          fullname: Anthon van der Neut\n          tmux:\n            session-name: test  # make sure this doesn't clash with\n                                # other sessions\n          loop:   # looping related defaults\n            # experiment with the following\n            wait: 10\n          # no more argument info to pass\n        ")

    def test_multi(self):
        round_trip("\n        # default arguments for the program\n        args:  # needed to prevent comment wrapping\n        # this should be your username\n          username: anthon\n          passwd: secret        # this is plaintext don't reuse\n                                # important/system passwords\n          fullname: Anthon van der Neut\n          tmux:\n            session-name: test  # make sure this doesn't clash with\n                                # other sessions\n          loop:   # looping related defaults\n            # experiment with the following\n            wait: 10\n          # no more argument info to pass\n        ")