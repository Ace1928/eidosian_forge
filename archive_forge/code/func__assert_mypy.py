from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def _assert_mypy(expect_success: bool, source_code: str) -> None:
    with NamedTemporaryFile(mode='w+t', delete=False) as config_file:
        config_file.write(MYPY_CONFIG)
    out, err, status = mypy.api.run(['--config-file', config_file.name, '-c', source_code])
    if status not in (0, 1):
        raise RuntimeError(f'Unexpected mypy error (status {status}):\n{indent(err, ' ' * 2)}')
    if expect_success:
        assert status == 0, f'Unexpected mypy failure (status {status}):\n{indent(out, ' ' * 2)}'
    else:
        assert status == 1, f'Unexpected mypy success: stdout: {out}\nstderr: {err}\n'