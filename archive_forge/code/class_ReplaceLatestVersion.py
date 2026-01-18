import argparse
from oslo_log import log as logging
from osc_lib import utils
from zunclient import api_versions
class ReplaceLatestVersion(argparse.Action):
    """Replaces `latest` keyword by last known version."""

    def __call__(self, parser, namespace, values, option_string=None):
        latest = values == '1.latest'
        if latest:
            values = '1'
        setattr(namespace, self.dest, values)