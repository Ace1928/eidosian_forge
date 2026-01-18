from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import pytest
from .. import Pooch
from ..processors import Unzip, Untar, Decompress
from .utils import pooch_test_url, pooch_test_registry, check_tiny_data, capture_log
def _unpacking_expected_paths_and_logs(archive, members, path, name):
    """
    Generate the appropriate expected paths and log message depending on the
    parameters for the test.
    """
    log_lines = ['Downloading']
    if archive == 'tiny-data':
        true_paths = {str(path / 'tiny-data.txt')}
        log_lines.append("Extracting 'tiny-data.txt'")
    elif archive == 'store' and members is None:
        true_paths = {str(path / 'store' / 'tiny-data.txt'), str(path / 'store' / 'subdir' / 'tiny-data.txt')}
        log_lines.append(f'{name}{name[-1]}ing contents')
    elif archive == 'store' and members is not None:
        true_paths = []
        for member in members:
            true_path = path / Path(*member.split('/'))
            if not str(true_path).endswith('tiny-data.txt'):
                true_path = true_path / 'tiny-data.txt'
            true_paths.append(str(true_path))
            log_lines.append(f"Extracting '{member}'")
        true_paths = set(true_paths)
    return (true_paths, log_lines)