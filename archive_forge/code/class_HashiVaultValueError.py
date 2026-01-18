from __future__ import absolute_import, division, print_function
import os
class HashiVaultValueError(ValueError):
    """Use in common code to raise an Exception that can be turned into AnsibleError or used to fail_json()"""