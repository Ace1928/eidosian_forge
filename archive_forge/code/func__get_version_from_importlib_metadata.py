import itertools
import operator
import sys
def _get_version_from_importlib_metadata(self):
    """Obtain a version from importlib or setup-time logic if missing.

        This will try to get the version of the package from the
        importlib_metadata record associated with the package, and if there
        is no such record falls back to the logic sdist would use.
        """
    try:
        distribution = importlib_metadata.distribution(self.package)
        result_string = distribution.version
    except importlib_metadata.PackageNotFoundError:
        from pbr import packaging
        result_string = packaging.get_version(self.package)
    return SemanticVersion.from_pip_string(result_string)