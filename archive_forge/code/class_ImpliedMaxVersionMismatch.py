import os_service_types
from keystoneauth1.exceptions import base
class ImpliedMaxVersionMismatch(ImpliedVersionMismatch):
    label = 'max_version'