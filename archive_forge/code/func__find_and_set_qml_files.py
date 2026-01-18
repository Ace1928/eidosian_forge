import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
def _find_and_set_qml_files(self):
    """Fetches all the qml_files in the folder and sets them if the
        field qml_files is empty in the config_dir"""
    if self.project_data:
        qml_files = self.project_data.qml_files
        for sub_project_file in self.project_data.sub_projects_files:
            qml_files.extend(ProjectData(project_file=sub_project_file).qml_files)
        self.qml_files = qml_files
    else:
        qml_files_temp = None
        source_file = Path(self.get_value('app', 'input_file')) if self.get_value('app', 'input_file') else None
        python_exe = Path(self.get_value('python', 'python_path')) if self.get_value('python', 'python_path') else None
        if source_file and python_exe:
            if not self.qml_files:
                qml_files_temp = list(source_file.parent.glob('**/*.qml'))
            if python_exe.parent.parent == source_file.parent:
                qml_files_temp = list(set(qml_files_temp) - set(python_exe.parent.parent.rglob('*.qml')))
            if len(qml_files_temp) > 500:
                if 'site-packages' in str(qml_files_temp[-1]):
                    raise RuntimeError('You are including a lot of QML files from a local virtual env. This can lead to errors in deployment.')
                else:
                    warnings.warn('You seem to include a lot of QML files. This can lead to errors in deployment.')
            if qml_files_temp:
                extra_qml_files = [Path(file) for file in qml_files_temp]
                self.qml_files.extend(extra_qml_files)
    if self.qml_files:
        self.set_value('qt', 'qml_files', ','.join([str(file.absolute().relative_to(self.project_dir)) for file in self.qml_files]))
        logging.info('[DEPLOY] QML files identified and set in config_file')