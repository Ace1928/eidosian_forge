from .roundtrip import YAML
import pytest  # NOQA
def Xtest_example_2_X():
    yaml = YAML()
    yaml.round_trip('\n    ')