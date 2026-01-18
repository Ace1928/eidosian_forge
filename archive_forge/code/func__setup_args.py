import glob
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Dict
from pkg_resources import parse_requirements
from setuptools import find_packages
def _setup_args() -> Dict[str, Any]:
    assistant = _load_assistant()
    about = _load_py_module('about', os.path.join(_PACKAGE_ROOT, '__about__.py'))
    version = _load_py_module('version', os.path.join(_PACKAGE_ROOT, '__version__.py'))
    long_description = assistant.load_readme_description(_PACKAGE_ROOT, homepage=about.__homepage__, version=version.version)
    return {'name': 'lightning-fabric', 'version': version.version, 'description': about.__docs__, 'author': about.__author__, 'author_email': about.__author_email__, 'url': about.__homepage__, 'download_url': 'https://github.com/Lightning-AI/lightning', 'license': about.__license__, 'packages': find_packages(where='src', include=['lightning_fabric', 'lightning_fabric.*']), 'package_dir': {'': 'src'}, 'long_description': long_description, 'long_description_content_type': 'text/markdown', 'include_package_data': True, 'zip_safe': False, 'keywords': ['deep learning', 'pytorch', 'AI'], 'python_requires': '>=3.8', 'setup_requires': ['wheel'], 'install_requires': assistant.load_requirements(_PATH_REQUIREMENTS, unfreeze='none' if _FREEZE_REQUIREMENTS else 'all'), 'entry_points': {'console_scripts': ['fabric = lightning_fabric.cli:_main']}, 'extras_require': _prepare_extras(), 'project_urls': {'Bug Tracker': 'https://github.com/Lightning-AI/lightning/issues', 'Documentation': 'https://pytorch-lightning.rtfd.io/en/latest/', 'Source Code': 'https://github.com/Lightning-AI/lightning'}, 'classifiers': ['Environment :: Console', 'Natural Language :: English', 'Development Status :: 4 - Beta', 'Intended Audience :: Developers', 'Topic :: Scientific/Engineering :: Artificial Intelligence', 'Topic :: Scientific/Engineering :: Information Analysis', 'Operating System :: OS Independent', 'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3.8', 'Programming Language :: Python :: 3.9', 'Programming Language :: Python :: 3.10', 'Programming Language :: Python :: 3.11']}