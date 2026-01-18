import itertools
import operator
import sys
def _get_version_from_pkg_resources(self):
    """Obtain a version from pkg_resources or setup-time logic if missing.

        This will try to get the version of the package from the pkg_resources
        This will try to get the version of the package from the
        record associated with the package, and if there is no such record
        importlib_metadata record associated with the package, and if there
        falls back to the logic sdist would use.

        is no such record falls back to the logic sdist would use.
        """
    import pkg_resources
    try:
        requirement = pkg_resources.Requirement.parse(self.package)
        provider = pkg_resources.get_provider(requirement)
        result_string = provider.version
    except pkg_resources.DistributionNotFound:
        from pbr import packaging
        result_string = packaging.get_version(self.package)
    return SemanticVersion.from_pip_string(result_string)