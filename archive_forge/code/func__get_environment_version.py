import argparse
from oslo_log import log as logging
from osc_lib import utils
from zunclient import api_versions
def _get_environment_version(default):
    env_value = utils.env('OS_CONTAINER_API_VERSION') or default
    latest = env_value == '1.latest'
    if latest:
        env_value = '1'
    return env_value