import platform
import sys
from .. import __version__
from ..utils.fixes import threadpool_info
from ._openmp_helpers import _openmp_parallelism_enabled
def _get_deps_info():
    """Overview of the installed version of main dependencies

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries

    """
    deps = ['pip', 'setuptools', 'numpy', 'scipy', 'Cython', 'pandas', 'matplotlib', 'joblib', 'threadpoolctl']
    deps_info = {'sklearn': __version__}
    from importlib.metadata import PackageNotFoundError, version
    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info