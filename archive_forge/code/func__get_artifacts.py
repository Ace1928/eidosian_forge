import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from project import (QmlProjectData, check_qml_decorators, is_python_file,
def _get_artifacts(self, file: Path) -> Tuple[List[Path], Optional[List[str]]]:
    """Return path and command for a file's artifact"""
    if file.suffix == '.ui':
        py_file = f'{file.parent}/ui_{file.stem}.py'
        return ([Path(py_file)], [UIC_CMD, os.fspath(file), '--rc-prefix', '-o', py_file])
    if file.suffix == '.qrc':
        py_file = f'{file.parent}/rc_{file.stem}.py'
        return ([Path(py_file)], [RCC_CMD, os.fspath(file), '-o', py_file])
    if file.suffix == '.py' and file in self._qml_module_sources:
        assert self._qml_module_dir
        qml_module_dir = os.fspath(self._qml_module_dir)
        json_file = f'{qml_module_dir}/{file.stem}{METATYPES_JSON_SUFFIX}'
        return ([Path(json_file)], [MOD_CMD, '-o', json_file, os.fspath(file)])
    if file.name.endswith(METATYPES_JSON_SUFFIX):
        assert self._qml_module_dir
        stem = file.name[:len(file.name) - len(METATYPES_JSON_SUFFIX)]
        qmltypes_file = self._qml_module_dir / f'{stem}.qmltypes'
        cpp_file = self._qml_module_dir / f'{stem}_qmltyperegistrations.cpp'
        cmd = [QMLTYPEREGISTRAR_CMD, '--generate-qmltypes', os.fspath(qmltypes_file), '-o', os.fspath(cpp_file), os.fspath(file)]
        cmd.extend(self._qml_project_data.registrar_options())
        return ([qmltypes_file, cpp_file], cmd)
    return ([], None)