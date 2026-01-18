from os import path
import sys
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from saharaclient.osc import utils
def _serialize_label_items(plugin):
    labels = {}
    pl_labels = plugin.get('plugin_labels', {})
    for label, data in pl_labels.items():
        labels['plugin: %s' % label] = data['status']
    vr_labels = plugin.get('version_labels', {})
    for version, version_data in vr_labels.items():
        for label, data in version_data.items():
            labels['plugin version %s: %s' % (version, label)] = data['status']
    labels = utils.prepare_data(labels, list(labels.keys()))
    return sorted(labels.items())