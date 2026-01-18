import _imp
import _io
import sys
import _warnings
import marshal
def _compile_bytecode(data, name=None, bytecode_path=None, source_path=None):
    """Compile bytecode as found in a pyc."""
    code = marshal.loads(data)
    if isinstance(code, _code_type):
        _bootstrap._verbose_message('code object from {!r}', bytecode_path)
        if source_path is not None:
            _imp._fix_co_filename(code, source_path)
        return code
    else:
        raise ImportError('Non-code object in {!r}'.format(bytecode_path), name=name, path=bytecode_path)