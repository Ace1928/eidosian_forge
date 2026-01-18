from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _PscInstanceConfig(alloydb_messages, allowed_consumer_projects=None):
    """Generates the PSC instance config for the instance."""
    psc_instance_config = alloydb_messages.PscInstanceConfig()
    psc_instance_config.allowedConsumerProjects = allowed_consumer_projects
    return psc_instance_config