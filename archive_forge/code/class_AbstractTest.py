import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
class AbstractTest:
    features = []
    features_groups = {}
    features_map = {}
    features_flags = set()

    def load_flags(self):
        pass

    def test_features(self):
        self.load_flags()
        for gname, features in self.features_groups.items():
            test_features = [self.cpu_have(f) for f in features]
            assert_features_equal(__cpu_features__.get(gname), all(test_features), gname)
        for feature_name in self.features:
            cpu_have = self.cpu_have(feature_name)
            npy_have = __cpu_features__.get(feature_name)
            assert_features_equal(npy_have, cpu_have, feature_name)

    def cpu_have(self, feature_name):
        map_names = self.features_map.get(feature_name, feature_name)
        if isinstance(map_names, str):
            return map_names in self.features_flags
        for f in map_names:
            if f in self.features_flags:
                return True
        return False

    def load_flags_cpuinfo(self, magic_key):
        self.features_flags = self.get_cpuinfo_item(magic_key)

    def get_cpuinfo_item(self, magic_key):
        values = set()
        with open('/proc/cpuinfo') as fd:
            for line in fd:
                if not line.startswith(magic_key):
                    continue
                flags_value = [s.strip() for s in line.split(':', 1)]
                if len(flags_value) == 2:
                    values = values.union(flags_value[1].upper().split())
        return values

    def load_flags_auxv(self):
        auxv = subprocess.check_output(['/bin/true'], env=dict(LD_SHOW_AUXV='1'))
        for at in auxv.split(b'\n'):
            if not at.startswith(b'AT_HWCAP'):
                continue
            hwcap_value = [s.strip() for s in at.split(b':', 1)]
            if len(hwcap_value) == 2:
                self.features_flags = self.features_flags.union(hwcap_value[1].upper().decode().split())