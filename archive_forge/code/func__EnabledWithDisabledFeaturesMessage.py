from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.os_config.troubleshoot import utils
def _EnabledWithDisabledFeaturesMessage(disabled_features):
    return 'Yes\nOS Config agent is enabled for this VM instance, but the following features are disabled:\n[' + disabled_features + '].\nSee https://cloud.google.com/compute/docs/manage-os#disable-features for instructions on how to make changes to this setting.'