import yaml
from openstack.orchestration.util import template_format
Takes a string and returns a dict containing the parsed structure.

    This includes determination of whether the string is using the
    YAML format.
    