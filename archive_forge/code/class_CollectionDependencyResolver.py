from __future__ import (absolute_import, division, print_function)
class CollectionDependencyResolver(Resolver):
    """A dependency resolver for Ansible Collections.

    This is a proxy class allowing us to abstract away importing resolvelib
    outside of the `ansible.galaxy.dependency_resolution` Python package.
    """