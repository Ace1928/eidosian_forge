import _imp
import _io
import sys
import _warnings
import marshal
def _cache_bytecode(self, source_path, bytecode_path, data):
    mode = _calc_mode(source_path)
    return self.set_data(bytecode_path, data, _mode=mode)