from __future__ import print_function
import sys
import textwrap
import pytest
from pathlib import Path
class TestRead:

    def test_multi_load(self):
        from srsly.ruamel_yaml import YAML
        yaml = YAML()
        yaml.load('a: 1')
        yaml.load('a: 1')

    def test_parse(self):
        from srsly.ruamel_yaml import YAML
        from srsly.ruamel_yaml.constructor import ConstructorError
        yaml = YAML(typ='safe')
        s = '- !User0 {age: 18, name: Anthon}'
        with pytest.raises(ConstructorError):
            yaml.load(s)
        yaml = YAML(typ='safe')
        for _ in yaml.parse(s):
            pass