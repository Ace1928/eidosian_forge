from __future__ import (absolute_import, division, print_function)

    Given a sequence of action/module names, returns a list of these names
    with the same names with the prefixes `ansible.builtin.` and
    `ansible.legacy.` added for all names that are not already FQCNs.
    